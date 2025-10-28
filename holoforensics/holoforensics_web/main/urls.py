from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('scenes/', views.scenes, name='scenes'),
    path('analysis/', views.analysis_dashboard, name='analysis_dashboard'),
    path('tools/', views.simple_tools, name='simple_tools'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
]
