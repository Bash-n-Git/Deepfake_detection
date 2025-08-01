from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('svm/', views.upload_svm, name='upload_svm'),
    path('reports_view/', views.reports_view, name='reports_view'),  # new
    path('siglip/', views.upload_siglip, name='upload_siglip'),
]
