import ast


def parse_string_to_tuple(s: str):
    """
    Parse actions in strings and convert to structured actions

    Args:
        s (str): The input string of actions

    Returns:
        tuple(List, List): The parameters. First list contains [x,y,z], second contains euler angles [roll, pitch, yaw]
    """
    if s is None:
        return None
    try:
        result = ast.literal_eval(s)
        if not isinstance(result, tuple) or len(result) != 2:
            return None

        def process_item(item):
            if item is None:
                return [0.0, 0.0, 0.0]
            if (
                isinstance(item, list)
                and len(item) == 3
                and all(isinstance(x, (int, float)) for x in item)
            ):
                return [float(x) for x in item]
            return None

        list1 = process_item(result[0])
        list2 = process_item(result[1])

        if list1 is None or list2 is None:
            return None

        return (list1, list2)

    except Exception:
        return None
