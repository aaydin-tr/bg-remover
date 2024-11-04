<br />
<div align="center">
  <h3 align="center">BG-Remover</h3>

  <p align="center">
    <br />
    <br />
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

## About The Project
This is a simple tool to remove the background from an image. It uses the [DIS](https://github.com/xuebinqin/DIS) model to detect the image segmentation and then select the main object in the image. The selected object is then pasted on a PNG image with a transparent background with RGBA values. Optionally, the user can also get caption of image using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model. And the user can also upscale the image to a specific megapixel size with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/)

## Installation
After setting up the .env file and choose the app port, you can run the app using the following methods:

#### Local
In order to run models on GPU, you need to have a CUDA compatible GPU and install the required dependencies. You can follow the official [PyTorch](https://pytorch.org/get-started/locally/) installation guide to install PyTorch with CUDA support. After installing PyTorch, you can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
python app.py
```
#### Dokcer

You can also run the app using Docker. Docker will automatically install the required dependencies and run the app. You don't need to install any dependencies manually. You can run the app using the following command:

```bash
docker-compose up
```

## Usage
You can use the app by sending a POST request to the `/remove-bg` endpoint with the following parameters:

- `url`: The URL of the image to remove the background from.
- `do_caption`: Whether to get the caption of the image or not. Default is `False`.
- `do_resize`: Whether to resize the image or not. Default is `False`.
- `megapixels`: The megapixel size to resize the image to. Default is `1.0` megapixels. (Only works if `do_resize` is `True`).
- `headers`: The headers to use when downloading the image. Default is an empty dictionary.

The app will return a JSON response with the following fields:

- `caption`: The caption of the image. This field will be `None` if `do_caption` is `False`.
- `resized`: The URL of the resized image. This field will be `None` if `do_resize` is `False`. (Base64 Encoded format)
- `cropped`: The URL of the cropped image with the background removed. (Base64 Encoded format)
- `crop_mask`: The URL of the crop mask image. (Base64 Encoded format)
- `error`: The error message if any error occurs. This field will be `None` if there is no error.


You can use the following CURL command to test the app:

```bash
curl -X POST "http://localhost:8000/remove-bg" -H "Content-Type: application/json" -d '{
    "url": "https://example.com/image.jpg",
    "do_caption": true,
    "do_resize": true,
    "megapixels": 2.0
}'
```

### Other Endpoints
- `/health`: Check the health of the app.
- `/`: The home page of the app.

