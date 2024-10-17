plugin = {
    "Name": "GroundingDINO-SAM",
    "Version": "0.1.0",
    "Author": "DeepMake",
    "Description": "Mask objects using a text prompt",
    "env": "groundingdino" 
}

config = {
    "dino_model_name": "GroundingDINO-SAM",
    "sam_model_name": "GroundingDINO-SAM",
    "dino_device": "cpu",
    "sam_device": "gpu",
    "model_urls": {
        "GroundingDINO": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "SAM": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
}
endpoints = {
    "get_mask": {
        "call": "execute",  # This should match the endpoint function name in your plugin.py
        "inputs": {
            "img": "Image",
            "prompt": "Text",
            "box_threshold": "Float(default=0.35, min=0.0, max=1.0, optional=true)",
            "text_threshold": "Float(default=0.25, min=0.0, max=1.0, optional=true)"
        },
        "outputs": {"output_mask": "Image"}
    }
}
