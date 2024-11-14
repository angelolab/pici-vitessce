import json
from collections.abc import Mapping
from typing import Any

from matplotlib.colors import is_color_like, to_rgb
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Reader
from pydantic import BaseModel, ConfigDict
from upath import UPath
from vitessce import (
    AnnDataWrapper,
    ImageOmeZarrWrapper,
    ObsSegmentationsOmeZarrWrapper,
    get_initial_coordination_scope_prefix,
)
from vitessce import CoordinationLevel as CL
from vitessce.config import (
    VitessceConfig,
    VitessceConfigCoordinationScope,
    VitessceConfigDataset,
    VitessceConfigView,
    VitessceConfigViewHConcat,
    VitessceConfigViewVConcat,
)


class VitessceViews(BaseModel):
    """Container for Vitessce configuration views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    view_type: str
    spatial_view: VitessceConfigView
    layerController_view: VitessceConfigView
    obsSets_view: VitessceConfigView
    featureList_view: VitessceConfigView
    obsSetSizes_view: VitessceConfigView


class ScopeConfig(BaseModel):
    """Configuration for coordination scopes in Vitessce."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    featureTypeScope: VitessceConfigCoordinationScope
    featureValueTypeScope: VitessceConfigCoordinationScope
    spatialSegmentationFilledScope: VitessceConfigCoordinationScope
    spatialSegmentationStrokeWidthScope: VitessceConfigCoordinationScope
    featureSelectionScope: VitessceConfigCoordinationScope
    obsSetSelectionScope: VitessceConfigCoordinationScope
    obsSetColorScope: VitessceConfigCoordinationScope
    obsColorEncodingScope: VitessceConfigCoordinationScope
    spatialChannelOpacityScope: VitessceConfigCoordinationScope


class OmeroChannelConfig(BaseModel):
    """Configuration for channels in an Omero image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    channel_name: str
    color: tuple[int, int, int]
    omero_index: int


class OmeroImageConfig(BaseModel):
    """Configuration for an Omero image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    channel_configs: list[OmeroChannelConfig] | list = []

    def __add__(self, other: OmeroChannelConfig):
        self.channel_configs.append(other)
        return self

    def __iter__(self):
        return iter(self.channel_configs)

    def __len__(self):
        return len(self.channel_configs)

    def __getitem__(self, index):
        return self.channel_configs[index]


def _validate_color(color) -> tuple[int, int, int]:
    if not is_color_like(color):
        raise ValueError(f"Color {color} is not a valid color")
    else:
        # Convert float RGB values (0-1) to integers (0-255)
        color = tuple(int(c * 255) for c in to_rgb(color))
        return color


def _get_omero_channel_index(omero_channels: list[Mapping[str, Any]], channel: str) -> int:
    for i, c in enumerate(omero_channels):
        if c["label"] == channel:
            return i
    raise ValueError(f"Channel {channel} not found in omero channels")


def _get_omero_channel_colors(store: UPath, channel_color_map: Mapping[str, Any]) -> OmeroImageConfig:
    reader = Reader(ZarrLocation(store))
    omero_channels = reader.zarr.root_attrs["omero"]["channels"]
    omero_channel_configs = OmeroImageConfig()
    for channel, color in channel_color_map.items():
        omero_index = _get_omero_channel_index(omero_channels, channel)
        omero_channel_configs += OmeroChannelConfig(
            channel_name=channel,
            color=_validate_color(color),
            omero_index=omero_index,
        )
    return omero_channel_configs


def add_vitessce_dataset(vc: VitessceConfig, fov: str, dataset_name: str) -> VitessceConfigDataset:
    """Add a datset to the Vitessce configuration.

    Parameters
    ----------
    vc
        The Vitessce configuration object.
    fov
        The name of the field of view.
    dataset_name
        The name of the dataset.

    Returns
    -------
        The dataset object.
    """
    dataset = vc.add_dataset(uid=f"{fov}-{dataset_name}", name=f"{fov}-{dataset_name}")
    anndata_path = (vc.base_dir / dataset_name / "whole_cell_table.zarr").relative_to(vc.base_dir).as_posix()
    segmentation_path = (vc.base_dir / dataset_name / "segmentation.ome.zarr").relative_to(vc.base_dir).as_posix()
    image_path = (vc.base_dir / dataset_name / "image.ome.zarr").relative_to(vc.base_dir).as_posix()

    anndata_wrapper = AnnDataWrapper(
        adata_path=anndata_path,
        obs_set_paths=[["obs/cell_meta_cluster_final_broad", "obs/cell_meta_cluster_final"]],
        obs_set_names=["Cell Cluster"],
        obs_labels_paths=["obs/label"],
        obs_labels_names=["obs/label"],
        obs_locations_path="obsm/spatial",
        obs_feature_matrix_path="X",
        coordination_values={
            "fileUid": f"{dataset_name}-table",
            "obsType": "cell",
            "featureType": "gene",
            "featureValueType": "expression",
        },
    )

    segmentation_wrapper = ObsSegmentationsOmeZarrWrapper(
        img_path=segmentation_path,
        obsTypesFromChannelNames=True,
        coordination_values={
            "fileUid": f"{dataset_name}-segmentation",
            "obsType": "cell",
        },
    )

    image_wrapper = ImageOmeZarrWrapper(
        img_path=image_path,
        coordination_values={"fileUid": f"{dataset_name}-image"},
    )

    dataset = dataset.add_object(anndata_wrapper).add_object(segmentation_wrapper).add_object(image_wrapper)

    return dataset


def create_view(vc: VitessceConfig, dataset: VitessceConfigDataset, view_type: str) -> VitessceViews:
    """Create a Vitessce view.

    Parameters
    ----------
    vc
        The Vitessce configuration object.
    dataset
        The Vitessce dataset object.
    view_type
        The type of view to create. Useful for differentiating between multiple sets of views.

    Returns
    -------
        The view object.
    """
    spatial_view = vc.add_view("spatialBeta", dataset=dataset).set_props(
        title=f"Cell Segmentation | {view_type.capitalize()}"
    )
    layerController_view = vc.add_view("layerControllerBeta", dataset=dataset).set_props(
        title=f"Channels | {view_type.capitalize()}"
    )
    obsSets_view = vc.add_view("obsSets", dataset=dataset).set_props(title=f"Cell Types | {view_type.capitalize()}")
    obsSetSizes_view = vc.add_view("obsSetSizes", dataset=dataset).set_props(
        title=f"Cell Cluster Sizes | {view_type.capitalize()}"
    )
    featureList_view = vc.add_view("featureList", dataset=dataset).set_props(
        title=f"Marker List | {view_type.capitalize()}", variablesLabelOverride="Marker"
    )
    return VitessceViews(
        view_type=view_type,
        spatial_view=spatial_view,
        layerController_view=layerController_view,
        obsSets_view=obsSets_view,
        featureList_view=featureList_view,
        obsSetSizes_view=obsSetSizes_view,
    )


def setup_scopes(vc: VitessceConfig, views: VitessceViews) -> ScopeConfig:
    """Setup coordination scopes for the Vitessce configuration.

    Parameters
    ----------
    vc
        The Vitessce configuration object.
    views
        The Vitessce views object.

    Returns
    -------
        The coordination scopes object.
    """
    # Create Scopes
    (featureTypeScope, featureValueTypeScope, spatialSegmentationFilledScope, spatialSegmentationStrokeWidthScope) = (
        vc.add_coordination(
            "featureType", "featureValueType", "spatialSegmentationFilled", "spatialSegmentationStrokeWidth"
        )
    )

    obsSetSelectionScope, obsSetColorScope, obsColorEncodingScope = vc.add_coordination(
        "obsSetSelection", "obsSetColor", "obsColorEncoding"
    )

    (featureSelectionScope,) = vc.add_coordination("featureSelection")

    (spatialChannelOpacityScope,) = vc.add_coordination("spatialChannelOpacity")

    # Set Scopes
    featureTypeScope.set_value("gene")
    featureValueTypeScope.set_value("expression")
    spatialSegmentationFilledScope.set_value(False)
    spatialSegmentationStrokeWidthScope.set_value(0.1)
    obsColorEncodingScope.set_value("cellSetSelection")
    spatialChannelOpacityScope.set_value(0.3)

    views.featureList_view.use_coordination(featureSelectionScope)
    views.obsSets_view.use_coordination(obsSetSelectionScope, obsSetColorScope, obsColorEncodingScope)
    views.obsSetSizes_view.use_coordination(obsSetSelectionScope, obsSetColorScope)

    return ScopeConfig(
        featureTypeScope=featureTypeScope,
        featureValueTypeScope=featureValueTypeScope,
        spatialSegmentationFilledScope=spatialSegmentationFilledScope,
        spatialSegmentationStrokeWidthScope=spatialSegmentationStrokeWidthScope,
        featureSelectionScope=featureSelectionScope,
        obsSetSelectionScope=obsSetSelectionScope,
        obsSetColorScope=obsSetColorScope,
        obsColorEncodingScope=obsColorEncodingScope,
        spatialChannelOpacityScope=spatialChannelOpacityScope,
    )


def link_views(
    vc: VitessceConfig,
    views: VitessceViews,
    scopes: ScopeConfig,
    vitessce_path: UPath,
    dataset_name: str,
    fov: str,
    channel_color_map: Mapping[str, Any],
):
    """Link views to coordination scopes.

    Parameters
    ----------
    vc
        The Vitessce configuration object.
    views
        The Vitessce views object.
    scopes
        The coordination scopes object.
    fov
        The name of the field of view.
    n_vars
        The number of variables in the dataset.
    """
    omero_zarr_path = vitessce_path / fov / dataset_name / "image.ome.zarr"
    omero_channel_configs = _get_omero_channel_colors(omero_zarr_path, channel_color_map)

    vc.link_views_by_dict(
        views=[
            views.spatial_view,
            views.layerController_view,
            views.obsSets_view,
            views.featureList_view,
            views.obsSetSizes_view,
        ],
        input_val={
            "imageLayer": CL(
                [
                    {
                        "fileUid": f"{views.view_type}-image",
                        "photometricInterpretation": "BlackIsZero",
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "featureValueColormapRange": [0, 1],
                        "spatialTargetResolution": None,
                        "imageChannel": CL(
                            [
                                {
                                    "spatialChannelOpacity": scopes.spatialChannelOpacityScope,
                                    "spatialTargetC": o.omero_index,
                                    "spatialChannelWindow": None,
                                    "spatialChannelColor": list(o.color),
                                    "spatialChannelVisible": True,
                                }
                                for o in omero_channel_configs
                            ]
                        ),
                    }
                ]
            ),
            "segmentationLayer": CL(
                [
                    {
                        "fileUid": f"{views.view_type}-segmentation",
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "segmentationChannel": CL(
                            [
                                {
                                    "obsType": "cell",
                                    "obsSetSelection": scopes.obsSetSelectionScope,
                                    "obsSetColor": scopes.obsSetColorScope,
                                    "obsColorEncoding": scopes.obsColorEncodingScope,
                                    "spatialTargetC": 0,
                                    "spatialChannelVisible": True,
                                    "featureType": scopes.featureTypeScope,
                                    "featureValueType": scopes.featureValueTypeScope,
                                    "featureSelection": scopes.featureSelectionScope,
                                    "spatialSegmentationFilled": scopes.spatialSegmentationFilledScope,
                                    "spatialSegmentationStrokeWidth": scopes.spatialSegmentationStrokeWidthScope,
                                    "legendVisible": True,
                                    "spatialChannelOpacity": 1.0,
                                }
                            ]
                        ),
                    }
                ]
            ),
        },
        meta=True,
        scope_prefix=get_initial_coordination_scope_prefix(f"{fov}-{views.view_type}", "image"),
    )


VitessceConcatView = VitessceConfigViewVConcat | VitessceConfigViewHConcat


def set_layout(
    views: VitessceViews,
) -> VitessceConcatView:
    """Set the layout of the Vitessce views.

    Parameters
    ----------
    views
        The Vitessce views object.

    Returns
    -------
        The layout of the Vitessce visualization as an object.
    """
    layout = views.spatial_view | (
        views.layerController_view / ((views.obsSets_view | views.obsSetSizes_view) / views.featureList_view)
    )
    return layout


def export_config(vc: VitessceConfig, vc_view: VitessceConfigView):
    """Export the Vitessce configuration to a JSON file.

    Parameters
    ----------
    vc
        The Vitessce configuration object.
    vc_view
        The Vitessce view object.
    """
    config_dir = vc.base_dir / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)

    with open(vc.base_dir / "configs" / f"{vc_view.view_type}-config.json", "w") as f:
        base_url = config_dir.relative_to(vc.base_dir).as_posix()
        json.dump(vc.to_dict(base_url=base_url), f, indent=4)
        # json.dump(vc.to_dict(base_url=vc.bae.resolve().absolute().as_posix()), f, indent=4)
