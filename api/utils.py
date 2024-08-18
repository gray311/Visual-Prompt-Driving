"""
Some util functions for text manipulation. Maybe deprecated in the future.
"""

import re
from typing import Tuple, List
from PIL import Image
import pdb
import io

import base64


def extract_image_list_from_md(text: str) -> Tuple[str, List[str]]:
    patterns = re.findall(MD_PIC_PATTERN, text)
    for i, pattern in enumerate(patterns):
        subs = f"[IMAGE_{i}]"
        text = text.replace(pattern, subs)
    img_list = list(map(lambda s: s[4:-1], patterns))
    return text, img_list


def open_image(image_path, force_blank_return=True):
    try:
        image = Image.open(image_path).convert("RGB")
    except:  # empty string or imageIOError
        if force_blank_return:
            image = Image.new("RGB", (24, 24), (0, 0, 0))  # black placeholder for input
        else:
            image = None
        if image_path != "":
            print(f"WARNING: Image path {image_path} not found. Using black placeholder.")
    return image


def encode_image_base64(vpath_or_bytesio, max_size=-1):
    if isinstance(vpath_or_bytesio, str):
        with open(vpath_or_bytesio, "rb") as image_file:
            if max_size > 0:
                image = Image.open(image_file)
                image.thumbnail((max_size, max_size))
                output_buffer = io.BytesIO()
                image.save(output_buffer, format='png')
                image_bytes = output_buffer.getvalue()
            else:
                image_bytes = image_file.read()

        return base64.b64encode(image_bytes).decode('utf-8')
    
    elif isinstance(vpath_or_bytesio, Image.Image):
        buffered = io.BytesIO()
        vpath_or_bytesio.save(buffered, format="PNG") 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image_PIL(image_path,max_size=-1):
    if max_size > 0:
        image = Image.open(image_path)
        image.thumbnail((max_size, max_size))
    else:
        image = Image.open(image_path)
    return image

