# Keep module-level handle available even after star-import name collisions.
import importlib as _importlib

from matplotlib.animation import FuncAnimation as FuncAnimation

from .axes import *
from .backend import mpl as mpl
from .backend import plt as plt
from .bimage import *
from .bimage_boundaries import *
from .colors2 import *
from .DraggablePatches import *
from .figure import close_all as close_all
from .figure import new_fig as new_fig
from .figure import save_fig as save_fig
from .figure import subplot_grid as subplot_grid
from .geometry import *
from .legend import *
from .Patches2 import *
from .plotting import *
from .specific import *
from .styles import *
from .threed import *
from .ticks import *
from .widgets import *

geometry = _importlib.import_module(".geometry", __name__)
del _importlib
