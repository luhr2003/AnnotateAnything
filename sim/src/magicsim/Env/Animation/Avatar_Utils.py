import carb
import omni.usd
from isaacsim.core.utils import prims
from pxr import Sdf, Usd, UsdGeom
from .settings import Settings, AssetPaths, PrimPaths

from omni.anim.people.scripts.custom_command.populate_anim_graph import (
    populate_anim_graph,
)


class CharacterUtil:
    def get_character_skelroot_by_root(character_prim):
        for prim in Usd.PrimRange(character_prim):
            if prim.GetTypeName() == "SkelRoot":
                return prim
        return None

    def get_character_name_by_index(i):
        if i == 0:
            return "Character"
        elif i < 10:
            return "Character_0" + str(i)
        else:
            return "Character_" + str(i)

    def get_character_name(character_prim):
        # For characters under /World/Characters, names are root names
        # For the rest, names are skelroot names
        prim_path = prims.get_prim_path(character_prim)
        if prim_path.startswith(PrimPaths.characters_parent_path()):
            return prim_path.split("/")[3]
        else:
            return prim_path.split("/")[-1]

    def get_character_pos(character_prim):
        matrix = omni.usd.get_world_transform_matrix(character_prim)
        return matrix.ExtractTranslation()

    def get_characters_root_in_stage(count=-1, count_invisible=False):
        stage = omni.usd.get_context().get_stage()
        character_list = []
        character_root_path = PrimPaths.characters_parent_path()

        if stage is None:
            return []

        folder_prim = stage.GetPrimAtPath(character_root_path)

        if (
            folder_prim is None
            or (not folder_prim.IsValid())
            or (not folder_prim.IsActive())
        ):
            return []

        children = folder_prim.GetAllChildren()
        for c in children:
            if len(character_list) >= count and count != -1:  # Get all if count is -1
                break
            if (
                count_invisible
                or UsdGeom.Imageable(c).ComputeVisibility() != UsdGeom.Tokens.invisible
            ):
                character_list.append(c)
        return character_list

    def get_characters_in_stage(count=-1, count_invisible=False):
        # Get a list of SkelRoot prims as characters
        stage = omni.usd.get_context().get_stage()
        character_root_path = PrimPaths.characters_parent_path()
        character_root = stage.GetPrimAtPath(character_root_path)
        character_list = []
        for prim in Usd.PrimRange(character_root):
            if len(character_list) >= count and count != -1:  # Get all if count is -1
                break
            if prim.GetTypeName() == "SkelRoot":
                if (
                    count_invisible
                    or UsdGeom.Imageable(prim).ComputeVisibility()
                    != UsdGeom.Tokens.invisible
                ):
                    character_list.append(prim)
        return character_list

    def load_default_biped_to_stage():
        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.characters_parent_path()
        parent_prim = stage.GetPrimAtPath(parent_path)
        if not parent_prim.IsValid():
            prims.create_prim(parent_path, "Xform")
            carb.log_info(f"Character parent prim is created at: {parent_path}.")
            parent_prim = stage.GetPrimAtPath(parent_path)

        biped_prim_path = PrimPaths.biped_prim_path()
        biped_prim = stage.GetPrimAtPath(biped_prim_path)

        if Settings.skip_biped_setup():
            carb.log_info("Skip setting up Biped.")
            return biped_prim

        if not biped_prim.IsValid():
            prim = prims.create_prim(
                biped_prim_path,
                "Xform",
                usd_path=AssetPaths.default_biped_asset_path(),
            )
            prim.GetAttribute("visibility").Set("invisible")
            carb.log_info(
                f"Biped prim is created at: {biped_prim_path}, usd_path = {AssetPaths.default_biped_asset_path()}."
            )
            biped_prim = stage.GetPrimAtPath(biped_prim_path)

        populate_anim_graph()

        return biped_prim

    def get_anim_graph_from_character(character_prim):
        for prim in Usd.PrimRange(character_prim):
            if prim.GetTypeName() == "AnimationGraph":
                return prim
        return None

    def get_default_biped_character():
        stage = omni.usd.get_context().get_stage()
        return stage.GetPrimAtPath(PrimPaths.biped_prim_path())

    def setup_animation_graph_to_character(
        character_skelroot_list: list, anim_graph_prim
    ):
        """
        Add animation graph for input characters in stage.
        Remove previous one if it exists
        """
        if anim_graph_prim is None or not anim_graph_prim.IsValid():
            carb.log_error("Unable to find an animation graph on stage.")
            return

        anim_graph_path = anim_graph_prim.GetPrimPath()
        paths = [Sdf.Path(prim.GetPrimPath()) for prim in character_skelroot_list]
        omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=paths)
        omni.kit.commands.execute(
            "ApplyAnimationGraphAPICommand",
            paths=paths,
            animation_graph_path=Sdf.Path(anim_graph_path),
        )

    def setup_python_scripts_to_character(
        character_skelroot_list: list, python_script_path
    ):
        """
        Add behavior script for input characters in stage.
        Remove previous one if it exists.
        """
        paths = [Sdf.Path(prim.GetPrimPath()) for prim in character_skelroot_list]
        omni.kit.commands.execute("RemoveScriptingAPICommand", paths=paths)
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=paths)
        for prim in character_skelroot_list:
            attr = prim.GetAttribute("omni:scripting:scripts")
            attr.Set([r"{}".format(python_script_path)])

    # Delete one character prim bt the given name
    def delete_character_prim(char_name):
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.characters_parent_path()):
            carb.log_error(
                str(PrimPaths.characters_parent_path()) + " is not a valid prim path"
            )
            return

        character_prim = stage.GetPrimAtPath(
            "{}/{}".format(PrimPaths.characters_parent_path(), char_name)
        )
        if character_prim and character_prim.IsValid() and character_prim.IsActive():
            prims.delete_prim(character_prim.GetPath())

    # Delete all character prims in the stage
    def delete_character_prims():
        """
        Delete previously loaded character prims. Also deletes the default skeleton and character animations if they
        were loaded using load_default_skeleton_and_animations. Also deletes state corresponding to characters
        loaded onto stage.
        """
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.characters_parent_path()):
            carb.log_error(
                str(PrimPaths.characters_parent_path()) + " is not a valid prim path"
            )
            return

        character_root_prim = stage.GetPrimAtPath(PrimPaths.characters_parent_path())
        if (
            character_root_prim
            and character_root_prim.IsValid()
            and character_root_prim.IsActive()
        ):
            for character_prim in character_root_prim.GetChildren():
                if (
                    character_prim
                    and character_prim.IsValid()
                    and character_prim.IsActive()
                ):
                    prims.delete_prim(character_prim.GetPath())
