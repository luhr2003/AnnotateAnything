"""Planner service modules (in-process servers / clients)."""

from typing import List, Optional, Sequence, Union

import torch


def _normalize_planner_devices(
    planner_devices: Optional[Union[str, torch.device, Sequence]],
    num_instances: int,
) -> List[torch.device]:
    """Resolve ``planner_devices`` into a per-instance list of ``torch.device``.

    - ``None``                    -> ``[cuda:0] * num_instances``
    - ``str`` / ``torch.device``  -> broadcast to all instances
    - list/tuple of length 1      -> broadcast to all instances
    - list/tuple of length N      -> must equal ``num_instances`` (asserted)
    """
    if planner_devices is None:
        planner_devices = "cuda:0"
    if isinstance(planner_devices, (str, torch.device)):
        return [torch.device(planner_devices) for _ in range(num_instances)]
    if isinstance(planner_devices, (list, tuple)):
        if len(planner_devices) == 1:
            return [torch.device(planner_devices[0]) for _ in range(num_instances)]
        assert len(planner_devices) == num_instances, (
            f"planner_devices length {len(planner_devices)} must equal "
            f"num_instances {num_instances}"
        )
        return [torch.device(d) for d in planner_devices]
    raise TypeError(
        f"planner_devices must be None/str/torch.device/list/tuple, "
        f"got {type(planner_devices)}"
    )
