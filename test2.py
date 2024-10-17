import os
import cv2
import torch
import io
import numpy as np
import torch
import uuid
import supervision as sv
import groundingdino.datasets.transforms as T
import huey
from huey.storage import FileStorage, SqliteStorage
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
#from groundingdino.util.inference import Model, predict, annotate
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from plugin import Plugin, fetch_image, store_image
from PIL import Image


#ALLOW THE DOWNLOAD FOR THE WEIGHTS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust the paths to match the nested structure
GROUNDING_DINO_CONFIG_PATH = os.path.join(CURRENT_DIR, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "GroundingDINO", "weights", "groundingdino_swint_ogc.pth")


# Verify paths
if not os.path.isfile(GROUNDING_DINO_CONFIG_PATH) or not os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH):
    raise FileNotFoundError("Grounding DINO configuration or checkpoint file not found.")

# Initialize the model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
#grounding_dino_model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)

# Indicate the SAM model is ready after successful setup
SAM_CHECKPOINT_PATH = os.path.join(CURRENT_DIR,"GroundingDINO", "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
SAM_ENCODER_VERSION = "vit_h"

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)
# FastAPI initialization for the plugin
app = FastAPI()

class DetectionRequest(BaseModel):
    image_id: str
    classes: List[str]
    box_threshold: float = 0.35
    text_threshold: float = 0.25

# storage = FileStorage("storage", path='huey_storage')
storage = SqliteStorage(name="storage", filename='huey_storage.db')

def store_image(img_data, img_id=None):
    if img_id is None:
      img_id = str(uuid.uuid4())
    if not isinstance(img_data, bytes):
        raise HTTPException(status_code=400, detail=f"Data must be stored in bytes")
    storage.put_data(img_id, img_data)
    return img_id

def fetch_image(img_id):
    img_data = storage.peek_data(img_id)
    if img_data == huey.constants.EmptyData:
        raise HTTPException(status_code=400, detail=f"No image found for id {img_id}")
    return img_data

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

@app.get("/image/get/{img_id}")
async def get_img(img_id: str):
    image_bytes = fetch_image(img_id)
    return Response(content=image_bytes, media_type="image/png")

@app.post("/image/upload")
async def upload_img(file: UploadFile = File(...)):
    try:
        print("Received file:", file.filename)  # Print the received filename
        img_data = await file.read()  # Make sure to read the file
        image_id = store_image(img_data)  # Store using the read data
        print("Stored image with ID:", image_id)  # Confirm stored image ID
        return {"status": "Success", "image_id": image_id}
    except Exception as e:
        print(f"Error in uploading image: {e}")  # Print any errors
        raise HTTPException(status_code=500, detail=str(e))

"""
@app.post("/detect")
async def detect_objects(request: DetectionRequest):
    try:
        image_bytes = fetch_image(request.image_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply similar transformations as load_image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_transformed, _ = transform(image, None)
        image_transformed = image_transformed.to(DEVICE)

        boxes, logits, phrases = predict(model=grounding_dino_model, image=image_transformed, caption=' '.join(request.classes),
                                         box_threshold=request.box_threshold, text_threshold=request.text_threshold)
        annotated_frame = annotate(np.array(image), boxes, logits, phrases)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        new_image_id = store_image(buffer.tobytes())
        return {"status": "Success", "output_img": new_image_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""


@app.post("/detect")
async def detect_objects(request: DetectionRequest):
    try:
        print("Fetching image...") # Debugging statement
        image_bytes = fetch_image(request.image_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Convert PIL Image to NumPy array (still in RGB)
        image_np = np.array(image)

        # Convert RGB to BGR for OpenCV
        image_bgr = image_np[:, :, ::-1]
        print(f"Image size after opening: {image.size}") # Debugging statement
        # Apply similar transformations as load_image
        #transform = T.Compose([
            #T.RandomResize([800], max_size=1333),
            #T.ToTensor(),
            #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #])
        #print("Applying transformations...") # Debugging statement
        #image_transformed, _ = transform(image, None)
        #image_transformed = image_transformed.to(DEVICE)
        print("Starting object detection...") # Debugging statement
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image_bgr,
            classes=enhance_class_name(class_names=request.classes),
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
            )
        print("Annotating image...") # Debugging statement
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
                f"{request.classes} {confidence:0.2f}" 
                for _, _, confidence, class_id, _ 
                in detections]
        annotated_frame = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, labels=labels)

        #boxes, logits, phrases = predict(model=grounding_dino_model, image=image_transformed, caption=' '.join(request.classes),
                                         #box_threshold=request.box_threshold, text_threshold=request.text_threshold)
        #annotated_frame = annotate(np.array(image), boxes, logits, phrases)
        print("Encoding annotated image...") # Debugging statement
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        new_image_id = store_image(buffer.tobytes())
        print(f"New image stored with id: {new_image_id}") # Debugging
        return {"status": "Success", "output_img": new_image_id}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print stack trace
        raise HTTPException(status_code=500, detail=str(e))


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

@app.post("/segment")
async def segment_objects(request: DetectionRequest):
    try:
        print("Fetching image...") # Debugging statement
        image_bytes = fetch_image(request.image_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Convert PIL Image to NumPy array (still in RGB)
        image_np = np.array(image)

        # Convert RGB to BGR for OpenCV
        image_bgr = image_np[:, :, ::-1]
        print(f"Image size after opening: {image.size}") # Debugging statement
        print("Starting object detection...") # Debugging statement
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image_bgr,
            classes=enhance_class_name(class_names=request.classes),
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
            )
        print("Annotating image...") # Debugging statement

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
                f"{request.classes} {confidence:0.2f}" 
                for _, _, confidence, class_id, _ 
                in detections]
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        #annotated_frame = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, labels=labels)

        _, buffer = cv2.imencode('.jpg', annotated_image)
        new_image_id = store_image(buffer.tobytes())
        return {"status": "Success", "output_img": new_image_id}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print stack trace
        raise HTTPException(status_code=500, detail=str(e))


