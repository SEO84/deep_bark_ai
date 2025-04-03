from PIL import Image
import numpy as np
import io

def process_image(image_bytes, target_size=(224,224)):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
