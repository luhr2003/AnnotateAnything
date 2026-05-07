"""
SquatDexGrasp: same as LocoDexGrasp but for SquatGraspEnv (bottle on floor).
Uses DexGrasp atomic skill; env termination: object_z > 0.2 counts as lifted.
"""

from magicsim.Collect.Command.LocoDexGrasp import LocoDexGrasp


class SquatDexGrasp(LocoDexGrasp):
    """Squat grasp: bottle on floor, same pipeline as LocoDexGrasp."""

    def update(self, info):
        result = super().update(info)
        if result is not None:
            result = dict(result)
            result["type"] = "SquatDexGrasp"
        return result
