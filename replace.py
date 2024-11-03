import base64
import io
import logging
import torch
from PIL import Image
from captioner.captioner import derive_caption
from image_utils import ensure_resolution
from segmenter.segmenter import segment

MEGAPIXELS = 1.0


def remove_background(
    id: str,
    original_image,
    megapixels=MEGAPIXELS,
    do_caption=False,
    do_resize=False,
):
    caption = None
    resized = None
    logging.info(f"ID: {id} Original size: {original_image.size}")

    if do_caption:
        logging.info(f"ID: {id} Captioning...")
        caption = derive_caption(original_image)
        logging.info(f"ID: {id} Caption: {caption}")
        torch.cuda.empty_cache()

    if do_resize:
        logging.info(f"ID: {id} Ensuring resolution ({megapixels}MP)...")
        resized = ensure_resolution(original_image, megapixels=megapixels)
        logging.info(f"ID: {id} Resized size: {resized.size}")
        torch.cuda.empty_cache()

    logging.info(f"ID: {id} Segmenting...")
    [cropped, crop_mask] = segment(resized)
    torch.cuda.empty_cache()

    result = {
        "caption": caption,
        "resized": pil_image_to_base64(resized),
        "cropped": pil_image_to_base64(cropped.convert("RGBA")),
        "crop_mask": pil_image_to_base64(crop_mask.convert("RGBA")),
    }

    return result


def pil_image_to_base64(image: Image.Image | None, format: str = "PNG") -> str:
    if image is None:
        return ""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = buffered.getvalue()
    base64_str = base64.b64encode(img_str).decode("utf-8")
    return base64_str
