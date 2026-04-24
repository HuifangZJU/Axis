from spatial_arithmetic_utils import *
# ============================================================
# CONFIG EXAMPLE (DLPFC z-axis analysis-mode structure)
# ============================================================
BIOLOGICAL_AXIS_NAME = "dlpfc_z"
GROUP_ID = 1
GRID_SIZE = 256

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
ANALYSIS_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    f"{BIOLOGICAL_AXIS_NAME}_group{GROUP_ID}_grid{GRID_SIZE}",
    "analysis",
)


aggregate_virtual_slices_max(
    input_dir=ANALYSIS_DIR,
    biological_axis_name=BIOLOGICAL_AXIS_NAME,
    reference_axis_value=0,   # choose whichever z you want to keep attrs from
    output_subdir="aggregation",
)
