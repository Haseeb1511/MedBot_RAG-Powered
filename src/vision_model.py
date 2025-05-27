import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage

def extract_text_from_image(pil_image, format="png"):
    buffered = BytesIO()
    pil_image.convert("RGB").save(buffered, format=format)
    buffered.seek(0)  # Reset the buffer's position to the beginning
    img_base64 = base64.b64encode(buffered.read()).decode()
    return f"data:image/{format.lower()};base64,{img_base64}"

def create_vision_message(pil_image,query):
    message = HumanMessage(content=[{
        "type":"text",
        "text":query},
        {
            "type":"image_url",
            "image_url":{"url":pil_image}
        }
        ])
    return message
