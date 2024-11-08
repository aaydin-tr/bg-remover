from transformers import pipeline
import logging

captioner = None
PROMPT = "The main subject of this picture is a"


def init(device):
    global captioner

    logging.info("Initializing captioner...")

    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        prompt=PROMPT,
        device=device,
    )


def derive_caption(image):
    result = captioner(image, max_new_tokens=20)
    raw_caption = result[0]["generated_text"]
    caption = raw_caption.lower().replace(PROMPT.lower(), "").strip()
    return caption
