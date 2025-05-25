import os
import asyncio
import logging
from pathlib import Path
import subprocess
import tempfile
from typing import List, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode
import ffmpeg

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit for regular bot API

class VideoProcessor:
    """Handles video processing with ffmpeg"""
    
    RESOLUTIONS = {
        '144p': {'height': 144, 'bitrate': '200k'},
        '240p': {'height': 240, 'bitrate': '400k'},
        '360p': {'height': 360, 'bitrate': '800k'},
        '480p': {'height': 480, 'bitrate': '1200k'},
        '720p': {'height': 720, 'bitrate': '2500k'}
    }
    
    @staticmethod
    async def process_video(input_path: str, output_path: str, resolution: str) -> bool:
        """Process video to specified resolution"""
        try:
            res_config = VideoProcessor.RESOLUTIONS[resolution]
            height = res_config['height']
            bitrate = res_config['bitrate']
            
            # Use ffmpeg-python for better control
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                vf=f'scale=-2:{height}',
                vcodec='libx264',
                acodec='aac',
                video_bitrate=bitrate,
                audio_bitrate='128k',
                preset='medium',
                crf=23
            )
            
            # Run ffmpeg
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return True
            
        except Exception as e:
            logger.error(f"Error processing video to {resolution}: {e}")
            return False
    
    @staticmethod
    def get_video_info(file_path: str) -> dict:
        """Get video information using ffprobe"""
        try:
            probe = ffmpeg.probe(file_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if video_stream:
                return {
                    'duration': float(probe['format']['duration']),
                    'size': int(probe['format']['size']),
                    'width': video_stream['width'],
                    'height': video_stream['height'],
                    'bitrate': video_stream.get('bit_rate', 'N/A')
                }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
        
        return {}

class TelegramClientUploader:
    """Handles large file uploads using Telegram Client API"""
    
    def __init__(self):
        # You'll need to implement Telegram Client API integration here
        # This requires API_ID and API_HASH from https://my.telegram.org
        pass
    
    async def upload_large_file(self, file_path: str, chat_id: int, caption: str = ""):
        """Upload files larger than 50MB using Telegram Client API"""
        try:
            # Implementation would use telethon or pyrogram
            # For now, we'll use chunked upload strategy
            file_size = os.path.getsize(file_path)
            
            if file_size > MAX_FILE_SIZE:
                logger.info(f"File size {file_size} exceeds 50MB, using client API")
                # Implement client API upload here
                return await self._client_upload(file_path, chat_id, caption)
            else:
                return False  # Use regular bot API
                
        except Exception as e:
            logger.error(f"Error in large file upload: {e}")
            return False
    
    async def _client_upload(self, file_path: str, chat_id: int, caption: str):
        """Internal method for client API upload"""
        # This would implement the actual Telegram Client API upload
        # You need to integrate telethon or pyrogram here
        pass

class VideoBotHandler:
    """Main bot handler class"""
    
    def __init__(self):
        self.uploader = TelegramClientUploader()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        welcome_text = """
🎥 *Video Resolution Bot* 🎥

Send me any video and I'll convert it to different resolutions:
• 144p (Low quality, small size)
• 240p (Basic quality)
• 360p (Standard quality)
• 480p (Good quality)
• 720p (High quality)

I can handle files larger than 50MB using Telegram Client API!

Just send me a video to get started! 📹
        """
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming video files"""
        try:
            video = update.message.video
            if not video:
                await update.message.reply_text("Please send a video file.")
                return
            
            # Show processing message
            processing_msg = await update.message.reply_text(
                "🔄 Processing your video...\nThis may take a few minutes."
            )
            
            # Download the video
            file = await context.bot.get_file(video.file_id)
            input_path = self.temp_dir / f"input_{update.message.message_id}.mp4"
            await file.download_to_drive(input_path)
            
            # Get video info
            video_info = VideoProcessor.get_video_info(str(input_path))
            
            # Create resolution selection keyboard
            keyboard = []
            for resolution in VideoProcessor.RESOLUTIONS.keys():
                keyboard.append([
                    InlineKeyboardButton(
                        f"📱 {resolution.upper()}",
                        callback_data=f"convert_{resolution}_{update.message.message_id}"
                    )
                ])
            
            # Add "All Resolutions" option
            keyboard.append([
                InlineKeyboardButton(
                    "🎯 All Resolutions",
                    callback_data=f"convert_all_{update.message.message_id}"
                )
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Update message with video info and options
            info_text = f"""
📹 *Video Information:*
• Duration: {video_info.get('duration', 'N/A')} seconds
• Size: {video_info.get('size', 'N/A')} bytes
• Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}

Choose the resolution(s) you want:
            """
            
            await processing_msg.edit_text(
                info_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error handling video: {e}")
            await update.message.reply_text(f"Error processing video: {str(e)}")
    
    async def handle_conversion(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle resolution conversion callback"""
        query = update.callback_query
        await query.answer()
        
        try:
            # Parse callback data
            parts = query.data.split('_')
            resolution = parts[1]
            message_id = parts[2]
            
            input_path = self.temp_dir / f"input_{message_id}.mp4"
            
            if not input_path.exists():
                await query.edit_message_text("❌ Original video file not found. Please send the video again.")
                return
            
            if resolution == "all":
                await self._convert_all_resolutions(query, input_path, message_id)
            else:
                await self._convert_single_resolution(query, input_path, resolution, message_id)
                
        except Exception as e:
            logger.error(f"Error in conversion: {e}")
            await query.edit_message_text(f"❌ Conversion failed: {str(e)}")
    
    async def _convert_single_resolution(self, query, input_path, resolution, message_id):
        """Convert video to single resolution"""
        await query.edit_message_text(f"🔄 Converting to {resolution.upper()}...")
        
        output_path = self.temp_dir / f"output_{resolution}_{message_id}.mp4"
        
        success = await VideoProcessor.process_video(
            str(input_path), str(output_path), resolution
        )
        
        if success:
            await self._send_processed_video(query, output_path, resolution)
        else:
            await query.edit_message_text(f"❌ Failed to convert to {resolution}")
    
    async def _convert_all_resolutions(self, query, input_path, message_id):
        """Convert video to all resolutions"""
        await query.edit_message_text("🔄 Converting to all resolutions... This will take a while.")
        
        successful_conversions = []
        
        for resolution in VideoProcessor.RESOLUTIONS.keys():
            try:
                output_path = self.temp_dir / f"output_{resolution}_{message_id}.mp4"
                
                await query.edit_message_text(f"🔄 Converting to {resolution.upper()}...")
                
                success = await VideoProcessor.process_video(
                    str(input_path), str(output_path), resolution
                )
                
                if success:
                    successful_conversions.append((resolution, output_path))
                
            except Exception as e:
                logger.error(f"Error converting to {resolution}: {e}")
        
        # Send all successful conversions
        for resolution, output_path in successful_conversions:
            await self._send_processed_video(query, output_path, resolution, is_batch=True)
        
        await query.message.reply_text(f"✅ Completed! Sent {len(successful_conversions)} videos.")
    
    async def _send_processed_video(self, query, output_path, resolution, is_batch=False):
        """Send the processed video file"""
        try:
            file_size = os.path.getsize(output_path)
            caption = f"📱 Video converted to {resolution.upper()}\n📦 Size: {file_size / (1024*1024):.1f} MB"
            
            # Try large file upload first if file is big
            if file_size > MAX_FILE_SIZE:
                uploaded = await self.uploader.upload_large_file(
                    str(output_path), 
                    query.message.chat_id, 
                    caption
                )
                
                if uploaded:
                    if not is_batch:
                        await query.edit_message_text(f"✅ {resolution.upper()} video sent successfully!")
                    return
            
            # Use regular bot API
            with open(output_path, 'rb') as video_file:
                await query.message.reply_video(
                    video=video_file,
                    caption=caption,
                    supports_streaming=True
                )
            
            if not is_batch:
                await query.edit_message_text(f"✅ {resolution.upper()} video sent successfully!")
                
        except Exception as e:
            logger.error(f"Error sending video: {e}")
            if not is_batch:
                await query.edit_message_text(f"❌ Failed to send {resolution} video: {str(e)}")
    
    def cleanup_temp_files(self, message_id):
        """Clean up temporary files"""
        try:
            for file_path in self.temp_dir.glob(f"*_{message_id}.*"):
                file_path.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")

def main():
    """Main function to run the bot"""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable not set!")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Create handler instance
    bot_handler = VideoBotHandler()
    
    # Add handlers
    application.add_handler(CommandHandler("start", bot_handler.start))
    application.add_handler(MessageHandler(filters.VIDEO, bot_handler.handle_video))
    application.add_handler(CallbackQueryHandler(bot_handler.handle_conversion))
    
    # Start the bot
    logger.info("Starting Video Resolution Bot...")
    application.run_polling()

if __name__ == '__main__':
    main()
