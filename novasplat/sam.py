import argparse
import logging
import os

from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

from huggingface_hub import login

from novasplat.masking import calculate_bbox_from_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    logging.getLogger(__name__).info(
        "No Hugging Face token found in environment; using anonymous access."
    )

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process input file and output directory."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input directory/file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-t", "--text_prompt", required=False, help="Text prompt for segmentation"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting mask files from %s", args.input)
    mask_filepaths = list(Path(args.input).glob("*.png"))

    logger.info("Initializing SAM3 processor")
    bpe_path = "thirdparty/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    sam_processor = Sam3Processor(model)

    for mask_filepath in mask_filepaths:
        mask_pixels = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        bbox = calculate_bbox_from_mask(mask_pixels)

        masks_dir = mask_filepath.parent
        rgbs_dir = masks_dir.parent / "rgb"
        rgb_filename = str.split(mask_filepath.stem, "_")[0] + ".jpg"
        rgb_filepath = rgbs_dir / rgb_filename
        output_filepath = output_dir / mask_filepath.name

        rgb_image = Image.open(rgb_filepath)
        width, height = rgb_image.size

        box_input_xywh = torch.tensor(bbox).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

        norm_box_cxcywh = (
            normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
        )

        inference_state = sam_processor.set_image(rgb_image)

        inference_state = sam_processor.set_text_prompt(
            state=inference_state, prompt=args.text_prompt
        )
        inference_state = sam_processor.add_geometric_prompt(
            state=inference_state, box=norm_box_cxcywh, label=True
        )

        mask = inference_state["masks"][0].squeeze(0).cpu()
        cv2.imwrite(str(output_filepath), (mask.numpy() * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
