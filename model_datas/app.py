from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

svm_model = joblib.load("svm_model.pkl")  # Upload this file to Colab
label_encoder = joblib.load("label_encoder.pkl")  # Upload this too

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = feature_extractor(image).squeeze().cpu().numpy()
        feat = features.flatten().reshape(1, -1)

        pred = svm_model.predict(feat)[0]
        probs = svm_model.predict_proba(feat)[0]
        result = label_encoder.inverse_transform([pred])[0]

        return JSONResponse({
            "prediction": result,
            "probabilities": dict(zip(label_encoder.classes_, probs.round(3).tolist()))
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
