from django.shortcuts import render
from django.http import HttpResponse

def scene_list(request):
    return HttpResponse("<h1>Scene List</h1><p>Coming soon...</p>")

def scene_detail(request, scene_id):
    return HttpResponse(f"<h1>Scene Detail: {scene_id}</h1><p>Coming soon...</p>")

def scene_results(request, scene_id):
    return HttpResponse(f"<h1>Scene Results: {scene_id}</h1><p>Coming soon...</p>")

def upload_page(request):
    return HttpResponse("<h1>Upload Page</h1><p>Coming soon...</p>")
