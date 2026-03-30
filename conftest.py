import pytest


@pytest.fixture(autouse=True)
def _close_mpl_figures():
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass
