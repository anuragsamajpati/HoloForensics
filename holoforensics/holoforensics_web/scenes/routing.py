from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/analysis/(?P<room_name>\w+)/$', consumers.AnalysisConsumer.as_asgi()),
    re_path(r'ws/progress/(?P<room_name>\w+)/$', consumers.ProgressConsumer.as_asgi()),
]
