from spatial_arithmetic_utils import *
# ============================================================
# CONFIG EXAMPLE (DLPFC z-axis analysis-mode structure)
# ============================================================
BIOLOGICAL_AXIS_NAME = "AD_time"
ANALYSIS_DIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis"


adjacent_virtual_differences(
    input_dir=ANALYSIS_DIR,
    biological_axis_name=BIOLOGICAL_AXIS_NAME,
    mode='forward',
    output_subdir="differential",

)

# interpolate_adjacent_virtual_slices(
#     input_dir=ANALYSIS_DIR,
#     biological_axis_name=BIOLOGICAL_AXIS_NAME,
#     n_pieces=5,
#     mode="forward",
#     output_subdir="interpolations",
#     attr_source="nearest",
#     include_endpoint=False,
# )
