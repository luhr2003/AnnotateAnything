"""Extended Physics Material with RigidBodyMaterialCfg properties support."""

from typing import Optional, Literal, Dict, Any
import carb
import isaacsim.core.utils.stage as stage_utils
from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema

from magicsim.Env.Utils.usd_schema import safe_set_attribute_on_schema


class PhysicsMaterial:
    """Extended Physics Material with RigidBodyMaterialCfg properties.

    This class extends the base PhysicsMaterial to support all properties from
    RigidBodyMaterialCfg, including friction/restitution combine modes and
    compliant contact properties.

    Args:
        prim_path: USD prim path for the material
        name: Material name (default "physics_material")
        static_friction: Static friction coefficient (default None)
        dynamic_friction: Dynamic friction coefficient (default None)
        restitution: Restitution coefficient (default None)
        friction_combine_mode: Friction combine mode - "average", "min", "multiply", "max" (default None)
        restitution_combine_mode: Restitution combine mode - "average", "min", "multiply", "max" (default None)
        compliant_contact_stiffness: Spring stiffness for compliant contact (default None)
        compliant_contact_damping: Damping coefficient for compliant contact (default None)
        config: Optional dict config to load properties from (takes precedence over individual args)
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "physics_material",
        static_friction: Optional[float] = 0.5,
        dynamic_friction: Optional[float] = 0.5,
        restitution: Optional[float] = 0.0,
        friction_combine_mode: Optional[
            Literal["average", "min", "multiply", "max"]
        ] = "average",
        restitution_combine_mode: Optional[
            Literal["average", "min", "multiply", "max"]
        ] = "average",
        compliant_contact_stiffness: Optional[float] = 0.0,
        compliant_contact_damping: Optional[float] = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._prim_path = prim_path

        stage = stage_utils.get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            carb.log_info(f"Physics Material Prim already defined at path: {prim_path}")
            self._material = UsdShade.Material(stage.GetPrimAtPath(prim_path))
        else:
            self._material = UsdShade.Material.Define(stage, prim_path)

        self._prim = stage.GetPrimAtPath(prim_path)

        # Apply UsdPhysics.MaterialAPI for basic properties
        if self._prim.HasAPI(UsdPhysics.MaterialAPI):
            self._material_api = UsdPhysics.MaterialAPI(self._prim)
        else:
            self._material_api = UsdPhysics.MaterialAPI.Apply(self._prim)

        # Apply PhysxSchema.PhysxMaterialAPI for extended properties
        self._physx_material_api = PhysxSchema.PhysxMaterialAPI(self._prim)
        if not self._physx_material_api:
            self._physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(self._prim)

        # Load from config if provided (takes precedence)
        if config:
            static_friction = config.get("static_friction", static_friction)
            dynamic_friction = config.get("dynamic_friction", dynamic_friction)
            restitution = config.get("restitution", restitution)
            friction_combine_mode = config.get(
                "friction_combine_mode", friction_combine_mode
            )
            restitution_combine_mode = config.get(
                "restitution_combine_mode", restitution_combine_mode
            )
            compliant_contact_stiffness = config.get(
                "compliant_contact_stiffness", compliant_contact_stiffness
            )
            compliant_contact_damping = config.get(
                "compliant_contact_damping", compliant_contact_damping
            )

        # Set basic properties via UsdPhysics.MaterialAPI
        if static_friction is not None:
            self._material_api.CreateStaticFrictionAttr().Set(static_friction)
        if dynamic_friction is not None:
            self._material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
        if restitution is not None:
            self._material_api.CreateRestitutionAttr().Set(restitution)

        # Set extended properties via PhysxSchema.PhysxMaterialAPI
        safe_set_attribute_on_schema(
            self._physx_material_api, "friction_combine_mode", friction_combine_mode
        )
        safe_set_attribute_on_schema(
            self._physx_material_api,
            "restitution_combine_mode",
            restitution_combine_mode,
        )
        safe_set_attribute_on_schema(
            self._physx_material_api,
            "compliant_contact_stiffness",
            compliant_contact_stiffness,
        )
        safe_set_attribute_on_schema(
            self._physx_material_api,
            "compliant_contact_damping",
            compliant_contact_damping,
        )

        return

    @property
    def prim_path(self) -> str:
        """Get the prim path.

        Returns:
            str: Prim path
        """
        return self._prim_path

    @property
    def prim(self) -> Usd.Prim:
        """Get the USD prim.

        Returns:
            Usd.Prim: USD prim
        """
        return self._prim

    @property
    def name(self) -> str:
        """Get the material name.

        Returns:
            str: Material name
        """
        return self._name

    @property
    def material(self) -> UsdShade.Material:
        """Get the USD shade material.

        Returns:
            UsdShade.Material: USD shade material
        """
        return self._material

    def set_dynamic_friction(self, friction: float) -> None:
        """Set dynamic friction.

        Args:
            friction: Dynamic friction coefficient
        """
        if self._material_api.GetDynamicFrictionAttr().Get() is None:
            self._material_api.CreateDynamicFrictionAttr().Set(friction)
        else:
            self._material_api.GetDynamicFrictionAttr().Set(friction)
        return

    def get_dynamic_friction(self) -> Optional[float]:
        """Get dynamic friction.

        Returns:
            Dynamic friction coefficient or None if not set
        """
        if self._material_api.GetDynamicFrictionAttr().Get() is None:
            carb.log_warn("A dynamic friction attribute is not set yet")
            return None
        else:
            return self._material_api.GetDynamicFrictionAttr().Get()

    def set_static_friction(self, friction: float) -> None:
        """Set static friction.

        Args:
            friction: Static friction coefficient
        """
        if self._material_api.GetStaticFrictionAttr().Get() is None:
            self._material_api.CreateStaticFrictionAttr().Set(friction)
        else:
            self._material_api.GetStaticFrictionAttr().Set(friction)
        return

    def get_static_friction(self) -> Optional[float]:
        """Get static friction.

        Returns:
            Static friction coefficient or None if not set
        """
        if self._material_api.GetStaticFrictionAttr().Get() is None:
            carb.log_warn("A static friction attribute is not set yet")
            return None
        else:
            return self._material_api.GetStaticFrictionAttr().Get()

    def set_restitution(self, restitution: float) -> None:
        """Set restitution.

        Args:
            restitution: Restitution coefficient
        """
        if self._material_api.GetRestitutionAttr().Get() is None:
            self._material_api.CreateRestitutionAttr().Set(restitution)
        else:
            self._material_api.GetRestitutionAttr().Set(restitution)
        return

    def get_restitution(self) -> Optional[float]:
        """Get restitution.

        Returns:
            Restitution coefficient or None if not set
        """
        if self._material_api.GetRestitutionAttr().Get() is None:
            carb.log_warn("A restitution attribute is not set yet")
            return None
        else:
            return self._material_api.GetRestitutionAttr().Get()

    def set_friction_combine_mode(
        self, mode: Literal["average", "min", "multiply", "max"]
    ) -> None:
        """Set friction combine mode.

        Args:
            mode: Friction combine mode
        """
        safe_set_attribute_on_schema(
            self._physx_material_api, "friction_combine_mode", mode
        )
        return

    def get_friction_combine_mode(self) -> Optional[str]:
        """Get friction combine mode.

        Returns:
            Friction combine mode or None if not set
        """
        attr = self._physx_material_api.GetFrictionCombineModeAttr()
        if attr and attr.Get() is not None:
            return attr.Get()
        return None

    def set_restitution_combine_mode(
        self, mode: Literal["average", "min", "multiply", "max"]
    ) -> None:
        """Set restitution combine mode.

        Args:
            mode: Restitution combine mode
        """
        safe_set_attribute_on_schema(
            self._physx_material_api, "restitution_combine_mode", mode
        )
        return

    def get_restitution_combine_mode(self) -> Optional[str]:
        """Get restitution combine mode.

        Returns:
            Restitution combine mode or None if not set
        """
        attr = self._physx_material_api.GetRestitutionCombineModeAttr()
        if attr and attr.Get() is not None:
            return attr.Get()
        return None

    def set_compliant_contact_stiffness(self, stiffness: float) -> None:
        """Set compliant contact stiffness.

        Args:
            stiffness: Spring stiffness for compliant contact
        """
        safe_set_attribute_on_schema(
            self._physx_material_api, "compliant_contact_stiffness", stiffness
        )
        return

    def get_compliant_contact_stiffness(self) -> Optional[float]:
        """Get compliant contact stiffness.

        Returns:
            Compliant contact stiffness or None if not set
        """
        attr = self._physx_material_api.GetCompliantContactStiffnessAttr()
        if attr and attr.Get() is not None:
            return attr.Get()
        return None

    def set_compliant_contact_damping(self, damping: float) -> None:
        """Set compliant contact damping.

        Args:
            damping: Damping coefficient for compliant contact
        """
        safe_set_attribute_on_schema(
            self._physx_material_api, "compliant_contact_damping", damping
        )
        return

    def get_compliant_contact_damping(self) -> Optional[float]:
        """Get compliant contact damping.

        Returns:
            Compliant contact damping or None if not set
        """
        attr = self._physx_material_api.GetCompliantContactDampingAttr()
        if attr and attr.Get() is not None:
            return attr.Get()
        return None
