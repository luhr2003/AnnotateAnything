import pickle
import base64


def encode_data(data):
    """
    Encode data using base64 encoding and pickle serialization.

    Args:
        data: The data to be encoded.

    Returns:
        str: The base64 encoded string representation of the data.
    """
    # Serialize the data using pickle
    serialized_data = pickle.dumps(data)

    # Encode the serialized data using base64
    encoded_data = base64.b64encode(serialized_data).decode("utf-8")

    return encoded_data


def decode_data(encoded_data):
    """
    Decode data from a base64 encoded string and deserialize it using pickle.

    Args:
        encoded_data (str): The base64 encoded string representation of the data.

    Returns:
        The original data.
    """
    # Decode the base64 encoded string
    if encoded_data is None:
        return None
    decoded_data = base64.b64decode(encoded_data.decode("utf-8"))

    # Deserialize the data using pickle
    data = pickle.loads(decoded_data)

    return data
