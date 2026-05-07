from magicsim.Env.Robot.Cfg.Base import RobotCfg
from magicsim.Env.Robot.Cfg.Manipulator.Franka import FrankaCfg
from magicsim.Env.Robot.Cfg.Manipulator.FrankaRobotiq import FrankaRobotiqCfg
from magicsim.Env.Robot.Cfg.Manipulator.UR10 import UR10Cfg

from magicsim.Env.Robot.Cfg.Manipulator.FrankaTactile import FrankaTactileCfg
from magicsim.Env.Robot.Cfg.Mobile.novaCarter import NovaCarterCfg
from magicsim.Env.Robot.Cfg.Mobile.leatherback import LeatherbackCfg
from magicsim.Env.Robot.Cfg.Manipulator.FrankaUMI import FrankaUMICfg
from magicsim.Env.Robot.Cfg.Manipulator.PiperX import PiperXCfg
from magicsim.Env.Robot.Cfg.Manipulator.ArxX5 import ArxX5Cfg
from magicsim.Env.Robot.Cfg.Manipulator.SO101 import SO101Cfg
from magicsim.Env.Robot.Cfg.Manipulator.Openarm import OpenarmCfg
from magicsim.Env.Robot.Cfg.Manipulator.Xarm7 import Xarm7Cfg
from magicsim.Env.Robot.Cfg.Manipulator.UR5e import UR5eCfg
from magicsim.Env.Robot.Cfg.Manipulator.UR10e import UR10eCfg
from magicsim.Env.Robot.Cfg.MobileManip.ridgebackFranka import RidgebackFrankaCfg
from magicsim.Env.Robot.Cfg.MobileManip.ridgebackSawyer import RidgebackSawyerCfg
from magicsim.Env.Robot.Cfg.MobileManip.vega1psharpa import Vega1pSharpaCfg
from magicsim.Env.Robot.Cfg.MobileManip.genie1 import Genie1Cfg
from magicsim.Env.Robot.Cfg.MobileManip.mobileX7s import MobileX7sCfg
from magicsim.Env.Robot.Cfg.MobileManip.lift2 import Lift2Cfg

from magicsim.Env.Robot.Cfg.DualManipulator.Xtrainer import XtrainerCfg
from magicsim.Env.Robot.Cfg.DualManipulator.DualPiper import DualPiperCfg
from magicsim.Env.Robot.Cfg.DualManipulator.DualArxX5 import DualArxX5Cfg
from magicsim.Env.Robot.Cfg.DualManipulator.DualSO101 import DualSO101Cfg
from magicsim.Env.Robot.Cfg.DualManipulator.DualFranka import DualFrankaCfg
from magicsim.Env.Robot.Cfg.DualManipulator.DualOpenarm import DualOpenarmCfg
from magicsim.Env.Robot.Cfg.Dexterous.FrankaXhand import FrankaXhandCfg
from magicsim.Env.Robot.Cfg.Humanoid.G1 import G1Cfg
from magicsim.Env.Robot.Cfg.Humanoid.G1_Sonic import G1_SonicCfg
from magicsim.Env.Robot.Cfg.Humanoid.G1_Dex1 import G1_Dex1Cfg
from magicsim.Env.Robot.Cfg.Humanoid.G1_FixedHand import G1_FixedHandCfg
from magicsim.Env.Robot.Cfg.Quadruped.Go2 import Go2Cfg

ROBOT_DICT: dict[str, type[RobotCfg]] = {
    "franka": FrankaCfg,
    "frankarobotiq": FrankaRobotiqCfg,
    "ur10": UR10Cfg,
    # "ridgebackfranka": RidgebackFrankaCfg,
    "franka_tactile": FrankaTactileCfg,
    "novacarter": NovaCarterCfg,
    "leatherback": LeatherbackCfg,
    "franka_umi": FrankaUMICfg,
    "piper_x": PiperXCfg,
    "arx_x5": ArxX5Cfg,
    "so101": SO101Cfg,
    "openarm": OpenarmCfg,
    "xarm7": Xarm7Cfg,
    "ur5e": UR5eCfg,
    "ur10e": UR10eCfg,
    "ridgebackfranka": RidgebackFrankaCfg,
    "ridgebacksawyer": RidgebackSawyerCfg,
    "vega1p_sharpa": Vega1pSharpaCfg,
    "genie1": Genie1Cfg,
    "mobile_x7s": MobileX7sCfg,
    "lift2": Lift2Cfg,
    "xtrainer": XtrainerCfg,
    "dual_piper": DualPiperCfg,
    "dual_arx_x5": DualArxX5Cfg,
    "dual_so101": DualSO101Cfg,
    "dual_franka": DualFrankaCfg,
    "dual_openarm": DualOpenarmCfg,
    "franka_xhand": FrankaXhandCfg,
    "g1_sonic": G1_SonicCfg,
    "g1": G1Cfg,
    "g1_mobile": G1Cfg,
    "g1_dex1": G1_Dex1Cfg,
    "g1_fixed_hand": G1_FixedHandCfg,
    "go2": Go2Cfg,
}  # Add your robot cfg here
