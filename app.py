import os
# Optional: suppress oneDNN info message
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ✅ Confirmation message
print("Model loaded successfully")

def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_path = "images/1.jpg"   # make sure image is inside images folder
    caption = generate_caption(image_path)
    print("Generated Caption:", caption)

    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
