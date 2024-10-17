from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust the paths to match the nested structure
GROUNDING_DINO_CONFIG_PATH = os.path.join(CURRENT_DIR, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "GroundingDINO", "weights", "groundingdino_swint_ogc.pth")

model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)
IMAGE_PATH = "dog.jpeg"
TEXT_PROMPT = "dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)
print(f"Loaded image source shape: {image_source.size}, type: {type(image_source)}")
print(f"Loaded image tensor shape: {image.shape}, type: {type(image)}")

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)
print("After prediction")

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
print("After annotation")

cv2.imwrite("annotated_image.jpg", annotated_frame)
print("Saved annotated image")