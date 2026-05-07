from isaacsim.core.utils import semantics
from pxr import Usd
import carb


class StageUtil:
    def open_stage(usd_path: str, ignore_unsave=True):
        if not Usd.Stage.IsSupportedFile(usd_path):
            raise ValueError("Only USD files can be loaded")
        import carb.settings
        import omni.kit.window.file

        IGNORE_UNSAVED_CONFIG_KEY = "/app/file/ignoreUnsavedStage"
        old_val = carb.settings.get_settings().get(IGNORE_UNSAVED_CONFIG_KEY)
        carb.settings.get_settings().set(IGNORE_UNSAVED_CONFIG_KEY, ignore_unsave)
        omni.kit.window.file.open_stage(
            usd_path, omni.usd.UsdContextInitialLoadSet.LOAD_ALL
        )
        carb.settings.get_settings().set(IGNORE_UNSAVED_CONFIG_KEY, old_val)

    # Set the xform transformation type to be Scale, Orient, Trans, and return the original order
    # NOTE::I am planning to move this part to the util extension, since the camera calibration require the same feature
    def set_xformOpType_SOT():
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        original_xform_order_setting = carb.settings.get_settings().get(
            xformoptype_setting_path
        )
        carb.settings.get_settings().set(
            xformoptype_setting_path, "Scale, Orient, Translate"
        )
        return original_xform_order_setting

    def recover_xformOpType(original_xform_order_setting):
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        carb.settings.get_settings().set(
            xformoptype_setting_path, original_xform_order_setting
        )

    def fetch_semantic_label(target_prim, target_semantic_type: str = "class"):
        """fetch first semantic label with target type from the prim"""
        semantic_label = None
        # fetch all sematic labels attached on the object
        semantic_label_dict = semantics.get_semantics(target_prim)
        for key, type_to_vlaue in semantic_label_dict.items():
            semantic_type, semantic_value = tuple(type_to_vlaue)
            # ignore the case difference
            if str(semantic_type).lower() == target_semantic_type.lower():
                semantic_label = semantic_value
                break
        return semantic_label
