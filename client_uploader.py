"""
Enhanced uploader using Telegram Client API for files > 50MB
"""
import os
import logging
from telethon import TelegramClient
from telethon.tl.types import DocumentAttributeVideo

logger = logging.getLogger(__name__)

class TelegramClientUploader:
    def __init__(self, api_id: str, api_hash: str, bot_token: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        self.client = None
    
    async def start_client(self):
        """Initialize and start the Telegram client"""
        self.client = TelegramClient('bot_session', self.api_id, self.api_hash)
        await self.client.start(bot_token=self.bot_token)
    
    async def upload_large_video(self, file_path: str, chat_id: int, caption: str = ""):
        """Upload large video files using Telegram Client API"""
        try:
            if not self.client:
                await self.start_client()
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Uploading large file: {file_size} bytes")
            
            # Get video duration and dimensions for attributes
            import ffmpeg
            try:
                probe = ffmpeg.probe(file_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                duration = int(float(probe['format']['duration']))
                width = video_stream['width'] if video_stream else 0
                height = video_stream['height'] if video_stream else 0
            except:
                duration = width = height = 0
            
            # Create video attributes
            attributes = [DocumentAttributeVideo(
                duration=duration,
                w=width,
                h=height,
                supports_streaming=True
            )]
            
            # Upload the file
            await self.client.send_file(
                chat_id,
                file_path,
                caption=caption,
                attributes=attributes,
                supports_streaming=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading large file: {e}")
            return False
    
    async def close(self):
        """Close the client connection"""
        if self.client:
            await self.client.disconnect()
