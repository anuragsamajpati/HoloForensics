from django.urls import path
from . import views

urlpatterns = [
    path('', views.scene_list, name='scene_list'),
    path('<str:scene_id>/', views.scene_detail, name='scene_detail'),
    path('<str:scene_id>/results/', views.scene_results, name='scene_results'),
    path('upload/', views.upload_page, name='upload_page'),
]
