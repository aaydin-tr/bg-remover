import os
import torch
import traceback
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from uuid import uuid4

from captioner.captioner import init as init_captioner
from upscaler.upscaler import init as init_upscaler
from segmenter.segmenter import init as init_segmenter
from replace import remove_background
import logging

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.info("Initializing models...")
init_captioner(device)
init_upscaler(device)
init_segmenter(device)
logging.info("Models initialized")

app = FastAPI()
output_dir = "./img_output"


class UploadResponse(BaseModel):
    caption: str | None
    resized: str | None
    cropped: str
    crop_mask: str


class UploadRequest(BaseModel):
    url: str
    do_caption: bool = False
    do_resize: bool = False
    megapixels: float = 1.0
    headers: dict = {}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(req: UploadRequest):
    if not req.url:
        raise HTTPException(status_code=400, detail="Missing URL parameter")

    if req.do_resize and req.megapixels <= 0:
        raise HTTPException(status_code=400, detail="Invalid megapixels parameter")
    try:
        response = requests.get(
            req.url,
            headers={
                "User-Agent": USER_AGENT,
                **req.headers,
            },
        )
        response.raise_for_status()

        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail=f"Error fetching image: {response.status_code}"
            )

        id = str(uuid4())
        with open(f"{output_dir}/{id}.png", "wb") as f:
            f.write(response.content)

        logging.info(f"ID: {id} Using device: {device}")
        return remove_background(
            id=id,
            original_image=Image.open(f"{output_dir}/{id}.png"),
            megapixels=req.megapixels,
            do_caption=req.do_caption,
            do_resize=req.do_resize,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        os.remove(f"{output_dir}/{id}.png")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    tb = traceback.format_exc()

    response = {
        "error": str(exc),
        "traceback": tb,
        "description": "An unexpected error occurred",
    }
    return JSONResponse(status_code=500, content=response)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
