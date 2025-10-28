import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser

class AnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'analysis_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        message_type = text_data_json.get('type', 'message')

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'analysis_message',
                'message': message,
                'message_type': message_type
            }
        )

    # Receive message from room group
    async def analysis_message(self, event):
        message = event['message']
        message_type = event['message_type']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message,
            'type': message_type
        }))

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'progress_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        progress = text_data_json.get('progress', 0)
        status = text_data_json.get('status', 'processing')
        files_processed = text_data_json.get('files_processed', 0)
        total_files = text_data_json.get('total_files', 0)

        # Send progress to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'progress_update',
                'progress': progress,
                'status': status,
                'files_processed': files_processed,
                'total_files': total_files
            }
        )

    # Receive progress update from room group
    async def progress_update(self, event):
        progress = event['progress']
        status = event['status']
        files_processed = event['files_processed']
        total_files = event['total_files']

        # Send progress to WebSocket
        await self.send(text_data=json.dumps({
            'progress': progress,
            'status': status,
            'files_processed': files_processed,
            'total_files': total_files
        }))
