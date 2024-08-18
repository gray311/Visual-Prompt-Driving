import os
import json

from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
import numpy as np
import random
import tempfile
from io import BytesIO
import cv2

from api import GeminiEvaluator, GPTEvaluator, system_message, user_message

random.seed(233)


if __name__ == "__main__":

    model_name = "gpt"

    if model_name == "gemini":
        agent = GeminiEvaluator(api_key="AIzaSyAr6OfqGdlxo0BuKDE_8gJvZf00Vd6TRH0")
    
    elif model_name == "gpt":
        agent = GPTEvaluator(api_key="")

    instruction = "Please change the lane and proceed quickly through the intersection ahead."
    image_file = "./outputs/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610412404.jpg"

    question = {
        "prompted_system_content": system_message,
        "prompted_content": user_message.format(instruction=instruction),
        "image_list": [image_file],
    }

    response = agent.generate_answer(question)
    print(response['prediction'])
