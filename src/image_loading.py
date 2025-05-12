import requests
import os
from PIL import Image   # pillow
import base64
from io import BytesIO

def get_image(url,file_name,extension):
    os.makedirs("content",exist_ok=True)
    content = requests.get(url).content
    #save image to gile
    file_path = f"content/{file_name}.{extension}"
    with open(file_path,"wb") as f:
        f.write(content)
    image = Image.open(file_path)
    image.show()
    return image

image_url = "https://earthshotprize.org/wp-content/uploads/2023/05/bee-on-flower.jpg"
pic = get_image(image_url,"cat","png")


def pil_image_to_base64_url(pil_image,format="png"):
    buffered = BytesIO()
    pil_image.save(buffered,format=format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_base64}" 

image = pil_image_to_base64_url(pic)