from magicsim.Collect.Command.Reach import Reach
from magicsim.Collect.Command.DualReach import DualReach
from magicsim.Collect.CameraCommand.GoTo import GoTo
from magicsim.Collect.Command.Grasp import Grasp
from magicsim.Collect.Command.LocoReach import LocoReach
from magicsim.Collect.Command.LocoDexGrasp import LocoDexGrasp
from magicsim.Collect.Command.LocoRetractReach import LocoRetractReach
from magicsim.Collect.Command.DualLocoReach import DualLocoReach
from magicsim.Collect.Command.DualLocoRetractReach import DualLocoRetractReach
from magicsim.Collect.Command.NavTo import NavTo
from magicsim.Collect.Command.Push import Push
from magicsim.Collect.Command.Wave import Wave
from magicsim.Collect.Command.MobileReach import MobileReach
from magicsim.Collect.Command.MobileDualReach import MobileDualReach
from magicsim.Collect.Command.MobileGrasp import MobileGrasp
from magicsim.Collect.Command.DexGrasp import DexGrasp
from magicsim.Collect.Command.OpenDrawer import OpenDrawer
from magicsim.Collect.Command.DexOpenDrawer import DexOpenDrawer
from magicsim.Collect.Command.MobileOpenDrawer import MobileOpenDrawer
from magicsim.Collect.Command.CloseDrawer import CloseDrawer
from magicsim.Collect.Command.MobileCloseDrawer import MobileCloseDrawer
from magicsim.Collect.Command.LocoOpenDoor import LocoOpenDoor
from magicsim.Collect.Command.SquatDexGrasp import SquatDexGrasp
from magicsim.Collect.Command.LocoDexBiGrasp import LocoDexBiGrasp
from magicsim.Collect.Command.Fling import Fling
from magicsim.Collect.Command.Fold import Fold
from magicsim.Collect.Command.LocoLift import LocoLift
from magicsim.Collect.Command.LocoBox import LocoBox
from magicsim.Collect.Command.BiGrasp import BiGrasp
from magicsim.Collect.Command.BiDexGrasp import BiDexGrasp
from magicsim.Collect.Command.Handover import Handover

STR2TASK = {
    "Reach": Reach,
    "DualReach": DualReach,
    "Grasp": Grasp,
    "NavTo": NavTo,
    "LocoReach": LocoReach,
    "LocoDexGrasp": LocoDexGrasp,
    "LocoRetractReach": LocoRetractReach,
    "DualLocoReach": DualLocoReach,
    "DualLocoRetractReach": DualLocoRetractReach,
    "Push": Push,
    "Wave": Wave,
    "MobileReach": MobileReach,
    "MobileDualReach": MobileDualReach,
    "MobileGrasp": MobileGrasp,
    "DexGrasp": DexGrasp,
    "OpenDrawer": OpenDrawer,
    "DexOpenDrawer": DexOpenDrawer,
    "MobileOpenDrawer": MobileOpenDrawer,
    "CloseDrawer": CloseDrawer,
    "MobileCloseDrawer": MobileCloseDrawer,
    "LocoOpenDoor": LocoOpenDoor,
    "SquatDexGrasp": SquatDexGrasp,
    "LocoDexBiGrasp": LocoDexBiGrasp,
    "Fling": Fling,
    "Fold": Fold,
    "LocoLift": LocoLift,
    "LocoBox": LocoBox,
    "BiGrasp": BiGrasp,
    "BiDexGrasp": BiDexGrasp,
    "Handover": Handover,
}


CAMERA_STR2TASK = {
    "GoTo": GoTo,
}
