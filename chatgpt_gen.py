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
        agent = GeminiEvaluator(api_key="")
    
    elif model_name == "gpt":
        agent = GPTEvaluator(api_key="")

    instruction = "Please change the lane and proceed quickly through the intersection ahead."
    image_file = "./outputs/result/000000.jpg"

    question = {
        "prompted_system_content": system_message,
        "prompted_content": user_message.format(instruction=instruction),
        "image_list": [image_file],
    }

    response = agent.generate_answer(question)
    print(response['prediction'])
