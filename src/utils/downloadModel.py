import os
import requests

MODEL_URL = "https://huggingface.co/H-Kansal/YouTubeSentimentAnalysis/resolve/main/dl_model.h5"

MODEL_PATH = os.path.join(
    "artifact",
    "modeltraining",
    "dl_model.h5"
)


def download_model():
    """
    Download model from Hugging Face
    """
    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Downloading model...")

    response = requests.get(MODEL_URL, stream=True)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:

        for chunk in response.iter_content(chunk_size=8192):

            if chunk:
                f.write(chunk)

    print("Download Complete.")