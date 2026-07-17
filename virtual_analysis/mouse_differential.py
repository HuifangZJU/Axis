from spatial_arithmetic_utils import *
# ============================================================
# CONFIG EXAMPLE (DLPFC z-axis analysis-mode structure)
# ============================================================
# BIOLOGICAL_AXIS_NAME = "AD_time"
# ANALYSIS_DIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis"
#
#
# adjacent_virtual_differences(
#     input_dir=ANALYSIS_DIR,
#     biological_axis_name=BIOLOGICAL_AXIS_NAME,
#     mode='forward',
#     output_subdir="differential",
#
# )

# interpolate_adjacent_virtual_slices(
#     input_dir=ANALYSIS_DIR,
#     biological_axis_name=BIOLOGICAL_AXIS_NAME,
#     n_pieces=5,
#     mode="forward",
#     output_subdir="interpolations",
#     attr_source="nearest",
#     include_endpoint=False,
# )


out_paths = pairwise_virtual_differences(
    input_dir="/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf",
    biological_axis_name="Disease_time",
    time_pairs=[(102.5,2.5), (105.7,5.7), (113.4, 13.4)],
    mode="forward",
    output_subdir="differences_pairs",
)