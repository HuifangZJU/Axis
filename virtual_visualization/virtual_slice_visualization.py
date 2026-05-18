
from visualization_utils import *
# --------------------
# Config
# --------------------
def get_Tg_time_para(mode):
    GROUP_ID = 0
    ORIG_SAMPLES = [0, 1, 2]
    # ORIG_SAMPLES = [3, 4, 5]
    ORIG_Z_UM = {0: 2.5, 1: 5.7, 2: 17.9}
    # ORIG_Z_UM = {3: 2.5, 4: 5.7, 5: 13.4}
    GENE = "Arc"
    # GENE = "axis_joint_domain"

    ROOT = "/media/huifang/data/registration/xenium/mouse_brain"

    NET_IN  = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ads")
    # GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"Disease_delta_grid256/{mode}/differential/interpolations")
    GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"Tg_time_grid256/{mode}")
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual.png")
    S_REAL = 1
    S_VIRTUAL = 1
    INVERT_Y = False
    return NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y

def get_WT_time_para(mode):
    GROUP_ID = 0
    ORIG_SAMPLES = [3, 4, 5]
    ORIG_Z_UM = {3: 2.5, 4: 5.7, 5: 13.4}
    GENE = "Arc"

    ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
    NET_IN  = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ads/WT")
    GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"WT_time_grid256/{mode}")
    GEN_PATTERN = os.path.join(GEN_DIR, "virtual_WT_time_*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual.png")
    S_REAL = 1
    S_VIRTUAL = 1
    INVERT_Y=False
    return NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,'WT'

def get_dlpfc_z_para(mode,GROUP_ID):
    ORIG_SAMPLES = [1, 2]
    ORIG_Z_UM = { 0: 0, 1: 10, 2: 310, 3: 320}
    GENE = "PCP4"
    # GENE = "MFGE8"
    # GENE = "confidence"
    # GENE = "layer_guess_reordered"
    #
    ROOT = "/media/huifang/data/registration/DLPFC/huifang"
    NET_IN = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ad")

    # GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid80_spatial1/{mode}")
    GEN_DIR = os.path.join(f"/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/spatialz/dense_reconstruction/group{GROUP_ID}/")
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual.png")
    # OUT_PNG = None
    S_REAL = 5
    S_VIRTUAL = 5
    INVERT_Y = True
    return NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,None

def get_scc_z_para(mode,GROUP_ID):
    ORIG_SAMPLES = [0, 1,2]
    ORIG_Z_UM = { 0: 0, 1: 10, 2:20}
    # GENE = 'KRT1'
    # ['KRT1','MMP10','PTHLH']
    GENE="original_clusters"

    ROOT = "/media/huifang/data/registration/SCC/huifang"
    NET_IN = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ad")

    # GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"scc_z_group{GROUP_ID}_grid40_spatial3_gene_weight/{mode}")
    GEN_DIR = f"/media/huifang/data/registration/SCC/huifang/virtual_tissue/spatialz/group{GROUP_ID}"
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual.png")
    S_REAL = 16
    S_VIRTUAL = 12
    INVERT_Y = True
    return NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,None

# for MODE in ["query","analysis","virtual_cf"]:
# for GROUP_ID in [1]:
#     for MODE in ["query"]:
#         NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,H5ID = get_dlpfc_z_para(MODE,GROUP_ID)
#         visualize_slices(NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,H5ID)
NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y = get_Tg_time_para('analysis')
visualize_slices(NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y)
# for GROUP_ID in [2,5,9,10]:
#     # for MODE in ["query","analysis","virtual_cf"]:
#     for MODE in ["analysis"]:
#         NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,H5ID = get_scc_z_para(MODE,GROUP_ID)
#         visualize_slices(NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,H5ID)