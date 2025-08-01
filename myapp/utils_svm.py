import requests
from PIL import Image
import io

def predict_with_svm(image_path):
    api_url = "https://156d-35-240-201-12.ngrok-free.app/predict/"  # ✅ Append /predict/
    image = Image.open(image_path).convert("RGB")
    
    # Convert image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    # Send to API
    files = {"file": ("image.jpg", buffered, "image/jpeg")}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probabilities = result["probabilities"]
        return prediction, probabilities
    else:
        raise Exception(f"API call failed: {response.text}")
