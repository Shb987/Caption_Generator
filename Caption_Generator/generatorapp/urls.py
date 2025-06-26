
from django.urls import path
from . import views

urlpatterns = [
       path('', views.home, name='home'),
       path('api/predict/', views.predict_caption_api, name='predict-caption'),
# path("api/predict/", views.predict_caption_api, name="predict-caption")


]
