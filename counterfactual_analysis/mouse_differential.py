from spatial_arithmetic_utils import *
# ============================================================
# CONFIG EXAMPLE (DLPFC z-axis analysis-mode structure)
# ============================================================
BIOLOGICAL_AXIS_NAME = "Disease_delta"
GRID_SIZE = 256

ROOT = "/media/huifang/data/registration/xenium/mouse_brain/"
ANALYSIS_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    f"{BIOLOGICAL_AXIS_NAME}_grid{GRID_SIZE}",
    "analysis/differential",
)


# adjacent_virtual_differences(
#     input_dir=ANALYSIS_DIR,
#     biological_axis_name=BIOLOGICAL_AXIS_NAME,
#     mode='inverse',
#     output_subdir="differential",
#
# )

interpolate_adjacent_virtual_slices(
    input_dir=ANALYSIS_DIR,
    biological_axis_name=BIOLOGICAL_AXIS_NAME,
    n_pieces=5,
    mode="forward",
    output_subdir="interpolations",
    attr_source="nearest",
    include_endpoint=False,
)
