from virtual_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
import os
import numpy as np


# =========================
# HELPERS
# =========================
def maybe_log1p(X):
    return matrix_log1p(X) if APPLY_LOG1P_BEFORE_GENERATION else X


def load_anchor_bank():
    """
    Load all anchors once. The selected pair is chosen later per query.
    """
    bank = {}
    line = sennet_pairs[GROUP_ID]
    coor_key = "aligned_combined"
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}.h5ad")
    xenium_coor = np.stack([
        xenium_gene_data.obs[f'x_{coor_key}'].values,
        xenium_gene_data.obs[f'y_{coor_key}'].values
    ], axis=1)
    bank[ANALYSIS_REFERENCE_ANCHOR_ID] = {
        "axis_value": AXIS_POSITIONS[ANALYSIS_REFERENCE_ANCHOR_ID],
        "coords": xenium_coor
    }
    codex_protein_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}.h5ad")
    codex_coor = np.stack([
        codex_protein_data.obs[f'x_{coor_key}'].values,
        codex_protein_data.obs[f'y_{coor_key}'].values
    ], axis=1)

    attrs = codex_protein_data.obs.loc[:,ATTRIBUTE_COLUMNS].copy().reset_index(drop=True)
    expr_raw = codex_protein_data.obs.loc[:,PROTEIN_COLUMNS].copy().reset_index(drop=True)
    expr_raw = np.asarray(expr_raw)

    bank[ANALYSIS_MOVING_ANCHOR_ID] = {
        "axis_value": AXIS_POSITIONS[ANALYSIS_MOVING_ANCHOR_ID],
        "coords": codex_coor,
        "expr_raw": np.asarray(expr_raw),
        "genes": np.asarray(PROTEIN_COLUMNS),
        "attrs": attrs,
    }
    return bank


def run_single_cell_generation():
    refer = anchor_bank[ANALYSIS_REFERENCE_ANCHOR_ID]
    grid_pts = refer["coords"]



    axis_value = AXIS_POSITIONS[ANALYSIS_MOVING_ANCHOR_ID]
    rec = anchor_bank[ANALYSIS_MOVING_ANCHOR_ID]

    expr_raw = rec["expr_raw"]
    genes = rec["genes"]
    attrs_raw = rec["attrs"]
    coords = rec["coords"]

    expr_raw = expr_raw - 0.4*expr_raw.min()
    expr = maybe_log1p(expr_raw)

    print(
        f"[GEN] {BIOLOGICAL_AXIS_NAME}={axis_value}"
        f"data anchors=({ANALYSIS_MOVING_ANCHOR_ID}), location anchors = ({ANALYSIS_REFERENCE_ANCHOR_ID})"
    )

    point_weights = np.full(expr.shape[0], 1, dtype=float)

    if USE_ATTRIBUTES:
        attrs_cat = attrs_raw
    else:
        attrs_cat = None

    ad_grid = generate_virtual_tissue_fixed_grid(
        coords=coords,
        X=expr,
        grid_pts=grid_pts,
        var_names=genes,
        attrs=attrs_cat,
        attr_schema=ATTRIBUTE_SCHEMA,
        grid_support_mask=None,
        point_weights=point_weights,
        spatial_bandwidth=SPATIAL_BANDWIDTH,
        use_gene_weight=False,
        knn=KNN,
        min_neighbors=MIN_NEIGHBORS,
        radius=RADIUS,
        chunk_size=64,
        fill_value=0.0,
    )

    # metadata
    ad_grid.uns["apply_log1p_before_generation"] = bool(APPLY_LOG1P_BEFORE_GENERATION)
    ad_grid.uns["analysis_reference_anchor_id"] = int(ANALYSIS_REFERENCE_ANCHOR_ID)

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
    out_path = os.path.join(
        OUT_DIR,
        f"virtual_{axis_tag}_{GROUP_ID}_{axis_value}.h5ad"
    )
    ad_grid.write_h5ad(out_path)
    print("  saved:", out_path)

    print("DONE. Output folder:", OUT_DIR)




# =========================
# CONFIG
# =========================
BIOLOGICAL_AXIS_NAME = "sennet"


AXIS_POSITIONS = {
    0: 'xenium',
    1: 'protein'
}

# analysis mode uses one fixed reference slice/grid
ANALYSIS_REFERENCE_ANCHOR_ID = 0
ANALYSIS_MOVING_ANCHOR_ID=1

H5AD_DIR = "/media/huifang/data/sennet/Axis_registered_data/"

# preprocessing / weighting
APPLY_LOG1P_BEFORE_GENERATION = False
# for query, to sue neighboring information
AXIS_WEIGHT_BANDWIDTH = None
SPATIAL_BANDWIDTH = 1


KNN = 5
MIN_NEIGHBORS = 1
RADIUS = None


# gene generation
PROTEIN_COLUMNS=['p53', 'p16', 'pH2AX', 'HSP47', 'PD1', 'PDL1', 'CD32b', 'SPP1', 'CD23']

# optional attributes
USE_ATTRIBUTES = True
ATTRIBUTE_COLUMNS = ['cell_type_update']
ATTRIBUTE_SCHEMA = {'cell_type_update': "categorical"}
ATTR_SPATIAL_BANDWIDTH = None

GENE_SUBSET=None

OUT_DIR = "/media/huifang/data/sennet/Axis_generated_proteins/"
os.makedirs(OUT_DIR, exist_ok=True)
data_pairs = open('/media/huifang/data/sennet/xenium_codex_pairs.txt')
sennet_pairs = data_pairs.readlines()

for GROUP_ID in range(len(sennet_pairs)):
    if GROUP_ID == 8 or GROUP_ID == 25 or GROUP_ID == 26:
        continue
    print(f"group id {GROUP_ID}")

    # =========================
    # LOAD ALL ANCHORS ONCE
    # =========================

    ANCHOR_IDS = [0, 1]
    anchor_bank = load_anchor_bank()
    run_single_cell_generation()
