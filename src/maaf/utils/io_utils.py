# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import numpy as np
from io import BytesIO

def ensure_json_serializable(value):
    """
    Recursively ensures all values passed in are json serializable
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.float, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.uint8, np.int32, np.int64, np.integer)):
        return int(value)
    elif isinstance(value, dict):
        new_dict = {}
        for k, v in value.items():
            new_dict[k] = ensure_json_serializable(v)
        return new_dict
    elif isinstance(value, list):
        new_list = []
        for element in value:
            new_list.append(ensure_json_serializable(element))
        return new_list
    else:
        return value

def pil_image_to_bytes(pim):
    """
    Converts PIL image to b64 string.
    :params PIL.Image pim:
        the PIL image we want to convert
    :returns str:
        Returns a string of the b64 encoded pixels
    """
    buffer = BytesIO()
    pim.save(buffer, format="JPEG")
    img_str= buffer.getvalue()
    #img_str = base64.b64encode(buffer.getvalue())
    return(img_str)
