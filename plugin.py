import os
import cv2
import torch
import io
import numpy as np
import psutil
import threading
import requests
from tqdm import tqdm
import asyncio
import supervision as sv
from huey.storage import FileStorage, SqliteStorage
from typing import List
from argparse import Namespace
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from plugin import Plugin, fetch_image, store_image
from PIL import Image
from .config import plugin, config, endpoints
import sys


# FastAPI initialization for the plugin
app = FastAPI()

storage = SqliteStorage(name="storage", filename='huey_storage.db')

# Global variable initialization
gd_plugin = None

def check_model():
    if 'gd_plugin' not in globals():
        set_model()

@app.get("/get_info/")
def plugin_info():
    check_model()
    return gd_plugin.plugin_info()

@app.get("/get_config/")
def get_config():
    check_model()
    return gd_plugin.get_config()

@app.put("/set_config/")
def set_config(update: dict):
    gd_plugin.set_config(update) # TODO: Validate config dict are all valid keys
    return gd_plugin.get_config()


@app.on_event("startup")
async def startup_event():
    global gd_plugin
    print("Starting up")
    set_model()
    # Start the model download and setup in a separate thread
    if gd_plugin is not None:
        threading.Thread(target=gd_plugin.set_model, daemon=True).start()

        # Asynchronously wait for the model to be ready
        while not gd_plugin.is_model_ready():
            await asyncio.sleep(1)  # Check every second

        print("Successfully started up")
        gd_plugin.notify_main_system_of_startup("True")


@app.get("/set_model/")
def set_model():
    global gd_plugin
    if gd_plugin is None:
        args = {"plugin": plugin, "config": config, "endpoints": endpoints}
        dino_model_name = config["dino_model_name"]
        sam_model_name = config["sam_model_name"]
        gd_plugin = GD(Namespace(**args))
        return {"status": "Success", "detail": f"Models set successfully to {dino_model_name}, {sam_model_name}"}


@app.get("/execute/{img}/{prompt}")
def execute(img: str, prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
    # Use the GD plugin to perform segmentation
    segmented_mask = gd_plugin.segment_objects(img, prompt, box_threshold, text_threshold)

    if segmented_mask:
        print("Combined RGB mask successfully processed and stored.")
        return {"status": "Success", "output_mask": segmented_mask}
    else:
        print("No RGB mask was processed.")
        return {"status": "Failed", "message": "No RGB mask detected or processed"}


def self_terminate():
    time.sleep(3)
    parent = psutil.Process(psutil.Process(os.getpid()).ppid())
    print(f"Killing parent process {parent.pid}")
    # os.kill(parent.pid, 1)
    # parent.kill()

@app.get("/shutdown/")  #Shutdown the plugin
def shutdown():
    threading.Thread(target=self_terminate, daemon=True).start()
    return {"success": True}


class GD(Plugin):
    """
    Prediction inference.
    """
    DOWNLOADING = "DOWNLOADING"
    READY = "READY"

    def __init__(self, arguments: "Namespace") -> None:
        super().__init__(arguments, plugin_name="Gsam")
        self.set_model()
        self.model_is_ready = False

    def is_model_ready(self):
        return self.model_is_ready
    
    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]
    
    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            mask = masks[index]
            print(f"Mask shape: {mask.shape}, data type: {mask.dtype}")  # Debugging statement
            result_masks.append(mask)
        return np.array(result_masks)
    
    def download_with_progress(self, url, destination):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    
    def set_model(self) -> None:
        """
        Asynchronously load given weights into model.
        """
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        WEIGHTS_DIR = os.path.join(CURRENT_DIR, "GroundingDINO", "weights")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)  # Ensure the weights directory exists
        # Define paths for the configuration and weights
        GROUNDING_DINO_CONFIG_PATH = os.path.join(CURRENT_DIR, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
        # Adjust the paths to match the nested structure
        # Indicate the SAM model is ready after successful setup
        SAM_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
        print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
        SAM_ENCODER_VERSION = "vit_h"
        # URLs for the weights
        GROUNDING_DINO_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        # Download GroundingDINO weights if not present
        if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
            print("Downloading GroundingDINO weights...")
            self.download_with_progress(GROUNDING_DINO_URL, GROUNDING_DINO_CHECKPOINT_PATH)

        # Download SAM weights if not present
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            print("Downloading SAM weights...")
            self.download_with_progress(SAM_URL, SAM_CHECKPOINT_PATH)


        # Verify paths
        if not os.path.isfile(GROUNDING_DINO_CONFIG_PATH) or not os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH):
            raise FileNotFoundError("Grounding DINO configuration or checkpoint file not found.")

        # Initialize the model
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {DEVICE}")
        if sys.platform == "darwin":
            DEVICE = torch.device('mps')
        global grounding_dino_model
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device='cpu')

        global sam
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
        global sam_predictor
        sam_predictor = SamPredictor(sam)
        # Indicate the model is ready after successful setup
        self.model_is_ready = True
        print(f"Model is ready: {self.model_is_ready}")  # Print the readiness status

    def segment_objects(self, img, classes, box_threshold, text_threshold):
        global grounding_dino_model
        print("Fetching image...")
        image_bytes = fetch_image(img)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        image_bgr = image_np[:, :, ::-1]
        classes = [classes]
        print(classes)
        print(f"Image size after opening: {image.size}")

        print("Starting object detection...")
        detections = grounding_dino_model.predict_with_classes(
            image=image_bgr,
            classes=self.enhance_class_name(class_names=classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        print(f"Detected objects: {detections}")

        print("Converting detections to masks...")
        detections.mask = self.segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        print(f"Masks generated: {len(detections.mask)}")

        if detections.mask is not None and len(detections.mask) > 0:
            combined_mask = np.zeros_like(detections.mask[0], dtype=np.uint8)
            for mask in detections.mask:
                binary_mask = mask.astype(np.uint8)
                combined_mask = np.bitwise_or(combined_mask, binary_mask * 255)  # Combine masks

            # Serialize the combined mask to bytes
            combined_mask_image = Image.fromarray(combined_mask)
            buffer = io.BytesIO()
            combined_mask_image.save(buffer, format="PNG")
            combined_mask_bytes = buffer.getvalue()
            combined_mask_image_id = store_image(combined_mask_bytes)
            print(f"Combined mask stored with image_id: {combined_mask_image_id}")
            return combined_mask_image_id
        else:
            print("No masks detected.")
            return None
    




