
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from visualization_utils import *
# --------------------
# Config
# --------------------
def get_mouse_time_para(group,mode):
    GROUP_ID = 0
    ORIG_SAMPLES = [0, 1, 2]
    # ORIG_SAMPLES = [3, 4, 5]

    ORIG_Z_UM = {0: 2.5, 1: 5.7, 2: 17.9}
    # ORIG_Z_UM = {3: 2.5, 4: 5.7, 5: 13.4}

    GENE = "Arc"

    ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
    NET_IN = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, f"h5ads/{group}")

    GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"{group}_time_grid256/{mode}")
    GEN_PATTERN = os.path.join(GEN_DIR, f"virtual_{group}_time_*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual_3d_horizontal.png")

    INVERT_Y = True
    GEN_USE_VALID_MASK = True
    USE_LOG1P = True

    S_REAL = 1
    S_VIRTUAL = 5

    # transparency control
    ALPHA_MIN = 0.2     # almost invisible near zero
    ALPHA_MAX_REAL = 0.95
    ALPHA_MAX_VIRTUAL = 0.85
    ALPHA_GAMMA = 1.6    # >1 makes low values fade more aggressively

    VMIN = 0.0
    VMAX_PCT_REAL = 0.995
    VMAX_PCT_VIRTUAL = 0.995
    VMAX_HARD_CAP_REAL = None
    VMAX_HARD_CAP_VIRTUAL = None

    # tissue coordinate display range
    X_LIM = (-50, 1050)
    Y_LIM = (-50, 1050)

    # Magnify slice spacing while preserving actual z ordering/ratios
    DEPTH_SCALE = 1.0

    # view
    ELEV = 1
    AZIM = -130
    INVERT_Y = False
    INVERT_X = True
    return GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X

def get_dlpfc_z_para(mode):
    GROUP_ID = 1
    ORIG_SAMPLES = [0, 1, 2, 3]
    ORIG_Z_UM = {0: 0, 1: 10, 2: 310, 3: 320}
    # GENE = "MOBP"
    GENE="MFGE8"
    # GENE = "layer_guess_reordered"

    ROOT = "/media/huifang/data/registration/DLPFC/huifang"
    NET_IN = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ad")

    GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid256/{mode}/aggregation")
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual_3d_horizontal.png")

    INVERT_Y = True
    GEN_USE_VALID_MASK = True
    USE_LOG1P = True

    S_REAL = 1
    S_VIRTUAL = 1

    # transparency control
    ALPHA_MIN = 0.0    # almost invisible near zero
    ALPHA_MAX_REAL = 0.5
    ALPHA_MAX_VIRTUAL = 1.0
    ALPHA_GAMMA = 1.6    # >1 makes low values fade more aggressively

    VMIN = 0.0
    VMAX_PCT_REAL = 0.995
    VMAX_PCT_VIRTUAL = 0.995
    VMAX_HARD_CAP_REAL = None
    VMAX_HARD_CAP_VIRTUAL = None

    # tissue coordinate display range
    X_LIM = (-50, 550)
    Y_LIM = (-50, 550)

    # Magnify slice spacing while preserving actual z ordering/ratios
    DEPTH_SCALE = 1

    # # view
    # ELEV = 0
    # AZIM = 0

    # view
    ELEV = 15
    AZIM = -130
    INVERT_Y=True
    INVERT_X = False
    return GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X


def get_scc_z_para(mode):
    GROUP_ID = 10
    ORIG_SAMPLES = [0, 1, 2]
    ORIG_Z_UM = {0: 0, 1: 10, 2: 2}
    GENE = "KRT1"
    # ['KRT1','MMP10','PTHLH']

    ROOT = "/media/huifang/data/registration/SCC/huifang"
    NET_IN = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ad")

    GEN_DIR = os.path.join(ROOT, "virtual_tissue", f"scc_z_group{GROUP_ID}_grid64/{mode}")
    GEN_PATTERN = os.path.join(GEN_DIR, "virtual_scc_z_*.h5ad")
    OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual_3d_horizontal.png")

    INVERT_Y = True
    GEN_USE_VALID_MASK = True
    USE_LOG1P = True

    S_REAL = 1
    S_VIRTUAL = 15

    # transparency control
    ALPHA_MIN = 0.0    # almost invisible near zero
    ALPHA_MAX_REAL = 0.95
    ALPHA_MAX_VIRTUAL = 1.
    ALPHA_GAMMA = 1.6    # >1 makes low values fade more aggressively

    VMIN = 0.0
    VMAX_PCT_REAL = 0.995
    VMAX_PCT_VIRTUAL = 0.995
    VMAX_HARD_CAP_REAL = None
    VMAX_HARD_CAP_VIRTUAL = None

    # tissue coordinate display range
    X_LIM = (-50, 550)
    Y_LIM = (-50, 550)

    # Magnify slice spacing while preserving actual z ordering/ratios
    DEPTH_SCALE = 1

    # view
    ELEV = 15
    AZIM = 130
    INVERT_Y=True
    return GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y


# GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X = get_mouse_time_para('Tg','analysis')
# plot_3d(GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X)

# GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y = get_scc_z_para('analysis')
# plot_3d(GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y)

GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X = get_dlpfc_z_para('analysis')
plot_3d(GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X)