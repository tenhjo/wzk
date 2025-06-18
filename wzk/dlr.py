import os
import platform


# TODO move this file to a private repo and think about how to replace the functionality

PLATFORM_IS_LINUX = platform.system() == "Linux"

USERNAME = os.path.expanduser("~").split(sep="/")[-1]
TENH_JO = "tenh_jo"
F_JUSTIN = "f_justin"

GITHUB_USER = "tenhjo"

userdir = "USERDIR"
homelocal = "/home_local"
home = "/home"

MAC = "mac"
DLR = "dlr"
GCP = "gcp"
GBK = "gbk"


__LOCATION2USERNAME_DICT = {MAC: "jote",
                            DLR: "tenh_jo",
                            GCP: "johannes_tenhumberg_gmail_com"}
__USERNAME2LOCATION_DICT = {"jote": MAC,
                            "tenh_jo": DLR,
                            "baeuml": DLR,
                            "bauml": DLR,
                            "f_justin": DLR,
                            "johannes_tenhumberg_gmail_com": GCP}


# --- Functions --------------------------------------------------------------------------------------------------------
def where_am_i():
    try:
        location = __USERNAME2LOCATION_DICT[USERNAME]
    except KeyError:
        location = DLR

    return location


def __wrapper_user(user=None):
    if user is None:
        user = USERNAME
    return user


def get_userdir(user):
    user = __wrapper_user(user=user)
    if user == F_JUSTIN:
        return f"/volume/projects/{F_JUSTIN}"
    else:
        return f"{home}/{user}/{userdir}"


def get_home(user):
    user = __wrapper_user(user=user)
    return f"{home}/{user}"


def get_homelocal(user, host=None):
    user = __wrapper_user(user=user)
    if host is None:
        return f"{homelocal}/{user}"
    else:
        return f"/net/{host}{homelocal}/{user}"


LOCATION = where_am_i()

USERDIR = get_userdir(user=USERNAME)  # Daily Back-up, relies on connection -> not for large Measurements
HOMELOCAL = get_homelocal(user=USERNAME)  # No Back-up, but fastest drive -> use for calculation
HOME = get_home(user=USERNAME)
USB = f"/var/run/media/{TENH_JO}/DLR-MA"

USERDIR_TENH = get_userdir(user=TENH_JO)
USERDIR_JUSTIN = f"{get_userdir(user=F_JUSTIN)}/packages/motion_planning"

__DIR_BASE_DICT = {DLR: f"{USERDIR_TENH}",
                   MAC: "/Users/jote/Documents/PhD",
                   GCP: "/home/johannes_tenhumberg_gmail_com",
                   GBK: "gs://tenh_jo"}
__DIR_BASE = __DIR_BASE_DICT[LOCATION]

# --- Data -------------------------------------------------------------------------------------------------------------
DIR_DATA = f"{__DIR_BASE}/data"
DIR_DATA_CAL = f"{DIR_DATA}/calibration"


# --- Paper ------------------------------------------------------------------------------------------------------------
DIR_PAPER = f"{__DIR_BASE}/paper"

humanoids21_elastic = DIR_PAPER + "/2021-humanoids-elastic"
humanoids22_rgb = DIR_PAPER + "/2022-humanoids-rgb"
humanoids23_contact = DIR_PAPER + "/2023-humanoids-contact"
tro24_calibration = DIR_PAPER + "/2024-tro-calibration"

iros22_planning = DIR_PAPER + "/2022-iros-planning"
humanoids23_ik = DIR_PAPER + "/2023-humanoids-ik"

tum24_diss = DIR_PAPER + "/2024-tum-diss"

# TRO23_Planning = "/23tro_Planning"


# -- Projects ----------------------------------------------------------------------------------------------------------
def get_dir_models(location, user):
    if user == "f_justin":
        directory = f"{USERDIR}/packages/motion_planning"
    else:
        __DIR_MODELS_DICT = {DLR: f"{USERDIR_TENH}/data/models",
                             MAC: f"{__DIR_BASE_DICT[MAC]}/data/mogen/Automatica2022",
                             GCP: "/home/johannes_tenhumberg_gmail_com/sdb/Automatica2022"}
        directory = __DIR_MODELS_DICT[location]

    return directory


DIR_MODELS = get_dir_models(location=LOCATION, user=USERNAME)


# --- DLR Remote Access ------------------------------------------------------------------------------------------------

# VNC
# On remote PC (i.e. pandia):
#   vncpasswd  -> set password (optional)
#   vncserver -geometry 1680x1050 -depth 24 -> start server, with scaled resolution for mac
#
# On local PC (i.e. my laptop):
#   ssh -l tenh_jo -L 5901:pandia:5901 ssh.robotic.dlr.de  -> set link port to the remote display
#   open VNC and connect to home: 127.0.0.1:1

# Connect2
# ssh -D 8080 -l tenh_jo ssh.robotic.dlr.de
