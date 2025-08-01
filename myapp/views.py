# deepfake_app/views.py

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils_svm import predict_with_svm
from .utils_siglip import predict_with_siglip
import os
import pandas as pd
from django.conf import settings
from django.core.files.storage import default_storage

def home(request):
    return render(request, 'deepfake_app/home.html')

def upload_svm(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        filepath = fs.path(filename)

        try:
            prediction, probabilities = predict_with_svm(filepath)
        except Exception as e:
            return render(request, 'deepfake_app/error.html', {'error': str(e)})
        scaled_data = {key: value * 100 for key, value in probabilities.items()}
        return render(request, 'deepfake_app/result_svm.html', {
            'filename': fs.url(filename),
            'prediction': prediction,
            'probabilities': scaled_data
        })
    return render(request, 'deepfake_app/upload_svm.html')

def upload_siglip(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        filepath = fs.path(filename)

        prediction = predict_with_siglip(filepath)
        probs = {"Real":float(prediction*100),"Fake":float(100-(prediction*100))}

        pred = max(probs,key=probs.get)
        print(pred)
        return render(request, 'deepfake_app/result_siglip.html', {
            'filename': fs.url(filename),
            'prediction': pred,
            'probabilities': probs
        })
    return render(request, 'deepfake_app/upload_siglip.html')

def reports_view(request):
    # Paths to images (in /media/)
    image1 = 'i1.png'  # Update with actual filenames
    image2 = 'svmi1.png'

    # Paths to CSV classification reports (generated in Colab or backend)
    svm_report_path = os.path.join(settings.MEDIA_ROOT, 'svm_classification_report.csv')
    siglip_report_path = os.path.join(settings.MEDIA_ROOT, 'siglig_classification_report.csv')

    # Read CSV reports as DataFrames
    svm_df = pd.read_csv(svm_report_path, index_col=0)
    siglip_df = pd.read_csv(siglip_report_path, index_col=0)

    # Convert DataFrames to HTML
    svm_html = svm_df.to_html(classes='table table-striped', border=0)
    siglip_html = siglip_df.to_html(classes='table table-striped', border=0)

    return render(request, 'deepfake_app/reports.html', {
        'image1': image1,
        'image2': image2,
        'svm_report': svm_html,
        'siglip_report': siglip_html
    })