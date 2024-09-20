# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from comfyui_enums import SAMPLERS, SCHEDULERS

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def width_height_from_aspect_ratio(self, aspect_ratio):
        return ASPECT_RATIOS[aspect_ratio]

    def update_workflow(self, workflow, **kwargs):
        width, height = self.width_height_from_aspect_ratio(kwargs["aspect_ratio"])

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        empty_latent_image = workflow["5"]["inputs"]
        empty_latent_image["width"] = width
        empty_latent_image["height"] = height
        empty_latent_image["batch_size"] = kwargs["num_outputs"]

        shift = workflow["61"]["inputs"]
        shift["width"] = width
        shift["height"] = height
        shift["max_shift"] = kwargs["max_shift"]
        shift["base_shift"] = kwargs["base_shift"]

        basic_scheduler = workflow["17"]["inputs"]
        basic_scheduler["scheduler"] = kwargs["scheduler"]
        basic_scheduler["steps"] = kwargs["num_inference_steps"]

        workflow["16"]["inputs"]["sampler_name"] = kwargs["sampler"]
        workflow["25"]["inputs"]["noise_seed"] = kwargs["seed"]
        workflow["60"]["inputs"]["guidance"] = kwargs["guidance_scale"]
        workflow["99"]["inputs"]["blocks"] = kwargs["flux_layers_to_patch"]

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image."
        ),
        flux_layers_to_patch: str = Input(
            description="Flux Dev layers to patch. A new line separated list of layers with values or a regular expression matching multiple layers, for example: 'double_blocks.0.img_mod.lin.weight=1.01' or 'attn=1.01'. See readme for examples.",
            default="",
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image in text-to-image mode. The size will always be 1 megapixel, i.e. 1024x1024 if aspect ratio is 1:1.",
            choices=list(ASPECT_RATIOS.keys()),
            default="1:1",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps. More steps can give more detailed images, but take longer.",
            ge=1,
            le=50,
            default=28,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
            ge=0,
            le=10,
            default=3,
        ),
        max_shift: float = Input(
            description="Maximum shift",
            ge=0,
            le=10,
            default=1.15,
        ),
        base_shift: float = Input(
            description="Base shift",
            ge=0,
            le=10,
            default=0.5,
        ),
        sampler: str = Input(
            description="Sampler",
            choices=SAMPLERS,
            default="euler",
        ),
        scheduler: str = Input(
            description="Scheduler",
            choices=SCHEDULERS,
            default="simple",
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        self.comfyUI.connect()
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            flux_layers_to_patch=flux_layers_to_patch,
            aspect_ratio=aspect_ratio,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_shift=max_shift,
            base_shift=base_shift,
            seed=seed,
            sampler=sampler,
            scheduler=scheduler,
        )

        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
