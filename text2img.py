import os
import openai
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

key = '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' 
client = openai.OpenAI(api_key=key)

def create_img(prompt):
    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
    )
    image_url = response.data[0].url
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


