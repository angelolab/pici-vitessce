import warnings

import anndata as ad
import natsort as ns
import numpy as np
import polars as pl
from tifffile import imread
from tqdm.auto import tqdm
from upath import UPath
from vitessce.data_utils import VAR_CHUNK_SIZE, multiplex_img_to_ome_zarr, optimize_adata, optimize_arr
from zarr import DirectoryStore, consolidate_metadata


def process_cell_table_to_anndata(
    cell_table_path: UPath,
    fovs: list[str],
    markers: list[str],
    obs_cols: list[str],
    obsm_cols: list[str],
    segmentation_dir: UPath,
    fov_category: str,
    vitessce_path: UPath,
    rename_markers: dict[str, str] | None = None,
) -> None:
    """
    Conver the cell table to AnnData objects per FOV.

    Process the cell table to create AnnData objects for specified fields of view (FOVs),
    specified markers, and specified observation variables.
    Export them to Zarr stores.

    Parameters
    ----------
    cell_table_path
        Path to the parquet file containing cell data.
    fovs
        List of field of view identifiers to process.
    markers
        List of marker columns to include in the AnnData's X attribute.
    obs_cols
        List of observation columns to include in the AnnData's `obs` attribute.
    obsm_cols
        List of observation metadata columns to include in the AnnData's `obsm` attribute.
    segmentation_dir
        Path to the directory containing segmentation images.
    fov_category
        The category of the field of view (e.g., immune, tumor).
    vitessce_path
        Base output path for saving Vitessce-compatible data.
    rename_markers, optional
        Optional dictionary to rename markers in the cell table.
    """
    # Read parquet files
    cell_table = pl.read_parquet(source=cell_table_path)
    if rename_markers is not None:
        cell_table = cell_table.rename(rename_markers)

    for (_fov,), fov_ct in tqdm(
        ns.natsorted(
            cell_table.lazy()
            .select(markers + obs_cols + obsm_cols)
            .cast(dtypes={"label": pl.Int16})
            .filter(pl.col("fov").is_in(fovs))
            .collect()
            .group_by("fov"),
            key=lambda g: str(g[0]),
        )
    ):
        _fov = str(_fov)
        fov_ct = fov_ct.sort("label")

        # Load unique labels from the segmentation image (filter on labels which are present in the image)
        n_unique_labels = len(np.unique(imread(segmentation_dir / f"{_fov}_whole_cell.tiff")))  # type: ignore
        fov_ct = fov_ct.lazy().filter(pl.col("label").is_in(range(1, n_unique_labels + 1))).collect()

        # Create AnnData object
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fov_adata = ad.AnnData(
                X=fov_ct[markers].to_pandas(),
                obs=fov_ct[obs_cols].to_pandas(),
                obsm={"spatial": fov_ct[obsm_cols].to_numpy()},  # type: ignore
            )
            fov_adata.strings_to_categoricals()

        # Optimize the AnnData object
        optimized_adata = optimize_adata(
            fov_adata,
            optimize_X=True,
        )

        # Define output path for Vitessce data
        vitessce_fov_path = vitessce_path / _fov / fov_category

        if not vitessce_fov_path.exists():
            vitessce_fov_path.mkdir(parents=True, exist_ok=True)

        # Save the optimized AnnData to the Zarr format
        store_path = DirectoryStore(vitessce_fov_path / "whole_cell_table.zarr")
        optimized_adata.write_zarr(store=store_path, chunks=(fov_adata.shape[0], VAR_CHUNK_SIZE))
        consolidate_metadata(store=store_path)


def convert_fovs_to_zarr(
    fovs: list[str],
    fov_category: str,
    fovs_dir: UPath,
    markers: list[str],
    vitessce_path: UPath,
    marker_colormap: dict[str, str] | None = None,
) -> None:
    """Converts multiple fields of view (folders of TIFFs) into Zarr Stores.

    Parameters
    ----------
    fovs
        The list of field of views to convert.
    fov_category
        The category of the field of view (e.g., immune, tumor).
    fovs_dir
        The directory containing all the field of views.
    markers
        The channels to convert. If `None`, all channels will be used.
    vitessce_path
        The output directory to place the converted field of views.
    marker_colormap, optional
        A colormap per channel. If `None`, colors will be auto-assigned.
    """
    if isinstance(fovs, str):
        fovs = [fovs]

    for fov in tqdm(fovs):
        channel_paths = [fovs_dir / fov / f"{c}.tiff" for c in ns.natsorted(markers)]
        fov_img = optimize_arr(arr=imread(files=channel_paths))

        if not (vitessce_fov_path := vitessce_path / fov / fov_category).exists():
            vitessce_fov_path.mkdir(parents=True, exist_ok=True)
        store_path = DirectoryStore(vitessce_fov_path / "image.ome.zarr")

        multiplex_img_to_ome_zarr(
            img_arr=fov_img,
            channel_names=markers,
            img_name=fov,
            axes="cyx",
            chunks=(1, 256, 256),
            channel_colors=marker_colormap,
            output_path=store_path,
        )
        consolidate_metadata(store=store_path)


def convert_segmentations_to_zarr(
    fovs: list[str],
    fov_category: str,
    segmentation_mask_suffixes: str | list[str],
    segmentation_dir: UPath,
    vitessce_path: UPath,
) -> None:
    """Converts TIFF cell segmentation masks or compartment segmentation masks to OME-ZARR files for multiple FOVs.

    Parameters
    ----------
    fovs
        A list of fields of view.
    fov_category
        The category of the field of view (e.g., immune, tumor).
    segmentation_mask_suffixes
        The suffixes of the segmentation masks to convert (e.g. "whole_cell", "nuclear")
    segmentation_dir
        The directory containing the specific segmentation masks.
    vitessce_path
        The output directory to place the converted segmentation masks.
    """
    if isinstance(segmentation_mask_suffixes, str):
        segmentation_mask_suffixes = [segmentation_mask_suffixes]

    for fov in tqdm(fovs):
        segmentation_mask_paths = ns.natsorted(
            [segmentation_dir / f"{fov}_{c}.tiff" for c in segmentation_mask_suffixes]
        )
        segmentation_array = optimize_arr(arr=imread(files=segmentation_mask_paths).astype("int").squeeze())  # type: ignore

        if not (vitessce_seg_path := vitessce_path / fov / fov_category / "segmentation.ome.zarr").exists():
            vitessce_seg_path.mkdir(parents=True, exist_ok=True)

        store_path = DirectoryStore(vitessce_seg_path)

        multiplex_img_to_ome_zarr(
            img_arr=segmentation_array,
            channel_names=ns.natsorted(segmentation_mask_suffixes),
            img_name=f"{fov}_segmentation",
            axes="cyx" if segmentation_array.ndim == 3 else "yx",
            chunks=(1, 256, 256) if segmentation_array.ndim == 3 else (256, 256),
            channel_colors=None,
            output_path=store_path,
        )
        consolidate_metadata(store=store_path)
