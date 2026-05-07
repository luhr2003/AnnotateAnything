import os

MAGICSIM_HOME = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MAGICSIM_ASSETS = os.path.join(MAGICSIM_HOME, "Assets")
MAGICSIM_CONF = os.path.join(MAGICSIM_HOME, "src/magicsim/Env/Conf")


try:
    import isaaclab
except ImportError:
    raise RuntimeError(
        "IsaacLab not found. Please follow the instructions in the README to set up the environment."
    )
ISAACLAB_HOME = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(isaaclab.__file__)))
    )
)
