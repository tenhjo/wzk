from .backend import mpl as mpl, plt as plt
from .figure import (new_fig as new_fig,
                     save_fig as save_fig,
                     subplot_grid as subplot_grid,
                     close_all as close_all)


from .axes import *  # noqa: F403
from .colors2 import *  # noqa: F403
from .legend import *  # noqa: F403
from .ticks import *  # noqa: F403
from .styles import *  # noqa: F403

from .threed import *  # noqa: F403
from .bimage import *  # noqa: F403
from .bimage_boundaries import *  # noqa: F403
from .plotting import *  # noqa: F403
from .geometry import *  # noqa: F403

from .Patches2 import *  # noqa: F403
from .DraggablePatches import *  # noqa: F403
from .widgets import *  # noqa: F403

from .specific import *  # noqa: F403

from matplotlib.animation import FuncAnimation as FuncAnimation

# Keep module-level handle available even after star-import name collisions.
import importlib as _importlib  # noqa: E402
geometry = _importlib.import_module(".geometry", __name__)  # noqa: E402,F401
del _importlib
