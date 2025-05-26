import os
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode
import ffmpeg

# Configure logging for Railway
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Railway and Bot configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PORT = int(os.getenv('PORT', 8000))

# File size limits
MAX_BOT_API_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CLIENT_API_SIZE = 2000 * 1024 * 1024  # 2GB for Telegram Client API

class VideoProcessor:
    """Enhanced video processor optimized for Railway"""
    
    RESOLUTIONS = {
        '144p': {'height': 144, 'bitrate': '150k', 'audio_bitrate': '64k'},
        '240p': {'height': 240, 'bitrate': '300k', 'audio_bitrate': '64k'},
        '360p': {'height': 360, 'bitrate': '600k', 'audio_bitrate': '96k'},
        '480p': {'height': 480, 'bitrate': '1000k', 'audio_bitrate': '128k'},
        '720p': {'height': 720, 'bitrate': '2000k', 'audio_bitrate': '128k'}
    }
    
    @staticmethod
    async def process_video(input_path: str, output_path: str, resolution: str) -> bool:
        """Process video with optimized settings for Railway"""
        try:
            res_config = VideoProcessor.RESOLUTIONS[resolution]
            height = res_config['height']
            video_bitrate = res_config['bitrate']
            audio_bitrate = res_config['audio_bitrate']
            
            logger.info(f"Processing video to {resolution} - Input: {input_path}")
            
            # Create ffmpeg stream with optimized settings
            stream = ffmpeg.input(input_path)
            
            # Video processing with efficient encoding
            stream = ffmpeg.output(
                stream,
                output_path,
                vf=f'scale=-2:{height}',
                vcodec='libx264',
                acodec='aac',
                video_bitrate=video_bitrate,
                audio_bitrate=audio_bitrate,
                preset='medium',  # Good balance of speed and compression
                crf=28,  # Higher CRF for smaller files
                maxrate=video_bitrate,
                bufsize=f'{int(video_bitrate[:-1]) * 2}k',
                movflags='faststart',  # Web optimization
                threads=2  # Limit threads for Railway
            )
            
            # Run with progress tracking
            process = ffmpeg.run_async(stream, overwrite_output=True, quiet=True)
            await asyncio.create_task(asyncio.to_thread(process.wait))
            
            # Verify output file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully processed video to {resolution}")
                return True
            else:
                logger.error(f"Output file is empty or doesn't exist: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video to {resolution}: {e}")
            return False
    
    @staticmethod
    def get_video_info(file_path: str) -> dict:
        """Get comprehensive video information"""
        try:
            probe = ffmpeg.probe(file_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            info = {
                'duration': round(float(probe['format'].get('duration', 0)), 2),
                'size': int(probe['format'].get('size', 0)),
                'bitrate': int(probe['format'].get('bit_rate', 0)) if probe['format'].get('bit_rate') else 0,
                'format': probe['format'].get('format_name', 'unknown')
            }
            
            if video_stream:
                info.update({
                    'width': video_stream.get('width', 0),
                    'height': video_stream.get('height', 0),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                    'video_codec': video_stream.get('codec_name', 'unknown')
                })
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'sample_rate': audio_stream.get('sample_rate', 0)
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'duration': 0, 'size': 0, 'width': 0, 'height': 0}

class TelegramClientUploader:
    """Handles large file uploads using Telegram Client API"""
    
    def __init__(self):
        self.client = None
        self._session_file = None
    
    async def initialize(self):
        """Initialize Telegram client for large file uploads"""
        try:
            if not API_ID or not API_HASH:
                logger.warning("API_ID or API_HASH not provided - large file upload disabled")
                return False
            
            from telethon import TelegramClient
            from telethon.tl.types import DocumentAttributeVideo
            
            # Use in-memory session for Railway
            self.client = TelegramClient('railway_bot_session', int(API_ID), API_HASH)
            await self.client.start(bot_token=BOT_TOKEN)
            logger.info("Telegram Client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram Client: {e}")
            return False
    
    async def upload_large_video(self, file_path: str, chat_id: int, caption: str = "", 
                                progress_callback=None) -> bool:
        """Upload large video files using Telegram Client API"""
        try:
            if not self.client:
                if not await self.initialize():
                    return False
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Uploading large file: {file_size / (1024*1024):.1f} MB")
            
            # Get video metadata for attributes
            try:
                probe = ffmpeg.probe(file_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                duration = int(float(probe['format'].get('duration', 0)))
                width = video_stream.get('width', 0) if video_stream else 0
                height = video_stream.get('height', 0) if video_stream else 0
            except:
                duration = width = height = 0
            
            # Import here to avoid issues if telethon not available
            from telethon.tl.types import DocumentAttributeVideo
            
            # Create video attributes
            attributes = [DocumentAttributeVideo(
                duration=duration,
                w=width,
                h=height,
                supports_streaming=True
            )]
            
            # Upload with progress tracking
            async def progress(current, total):
                if progress_callback:
                    percent = (current / total) * 100
                    await progress_callback(f"Uploading: {percent:.1f}%")
            
            await self.client.send_file(
                chat_id,
                file_path,
                caption=caption,
                attributes=attributes,
                supports_streaming=True,
                progress_callback=progress if progress_callback else None
            )
            
            logger.info("Large file uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading large file: {e}")
            return False
    
    async def close(self):
        """Close client connection"""
        try:
            if self.client:
                await self.client.disconnect()
                logger.info("Telegram Client disconnected")
        except Exception as e:
            logger.error(f"Error closing client: {e}")

class VideoBotHandler:
    """Main bot handler optimized for Railway"""
    
    def __init__(self):
        self.uploader = TelegramClientUploader()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="videobot_"))
        self.processing_jobs = {}  # Track active processing jobs
        
        # Initialize uploader
        asyncio.create_task(self.uploader.initialize())
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command"""
        user = update.effective_user
        welcome_text = f"""
🎥 *Welcome {user.first_name}!* 🎥

🚀 *Railway Video Resolution Bot*

✨ *Features:*
• Convert videos to multiple resolutions
• Support for files up to 2GB (using Client API)
• Optimized compression for smaller file sizes
• Batch processing available

📱 *Available Resolutions:*
• 144p - Ultra compressed
• 240p - Low quality  
• 360p - Standard quality
• 480p - Good quality
• 720p - High definition

🔥 *Smart File Handling:*
• Files ≤50MB: Instant bot API
• Files >50MB: Advanced client API
• Automatic quality optimization

Just send me any video to get started! 📹
        """
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced video handler with better error handling"""
        try:
            video = update.message.video or update.message.document
            if not video:
                await update.message.reply_text("❌ Please send a video file.")
                return
            
            # Check file size limits
            file_size = getattr(video, 'file_size', 0)
            if file_size > MAX_CLIENT_API_SIZE:
                await update.message.reply_text(
                    f"❌ File too large! Maximum size: {MAX_CLIENT_API_SIZE / (1024*1024*1024):.1f}GB"
                )
                return
            
            # Show initial processing message
            status_msg = await update.message.reply_text(
                "🔄 *Processing your video...*\n"
                "📥 Downloading file...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Download the video
            try:
                file = await context.bot.get_file(video.file_id)
                input_path = self.temp_dir / f"input_{update.message.message_id}_{int(time.time())}.mp4"
                
                await status_msg.edit_text(
                    "🔄 *Processing your video...*\n"
                    f"📥 Downloading {file_size / (1024*1024):.1f}MB...",
                    parse_mode=ParseMode.MARKDOWN
                )
                
                await file.download_to_drive(input_path)
                
            except Exception as e:
                await status_msg.edit_text(f"❌ Download failed: {str(e)}")
                return
            
            # Get video information
            await status_msg.edit_text(
                "🔄 *Processing your video...*\n"
                "📊 Analyzing video...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            video_info = VideoProcessor.get_video_info(str(input_path))
            
            # Create enhanced resolution keyboard
            keyboard = []
            for resolution in VideoProcessor.RESOLUTIONS.keys():
                estimated_size = self._estimate_file_size(video_info, resolution)
                size_text = f"({estimated_size})"
                
                keyboard.append([
                    InlineKeyboardButton(
                        f"📱 {resolution.upper()} {size_text}",
                        callback_data=f"convert_{resolution}_{update.message.message_id}_{int(time.time())}"
                    )
                ])
            
            # Add batch options
            keyboard.extend([
                [InlineKeyboardButton(
                    "🎯 All Resolutions",
                    callback_data=f"convert_all_{update.message.message_id}_{int(time.time())}"
                )],
                [InlineKeyboardButton(
                    "⚡ Mobile Pack (144p+360p+480p)",
                    callback_data=f"convert_mobile_{update.message.message_id}_{int(time.time())}"
                )]
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Enhanced video info display
            duration_str = f"{int(video_info['duration'] // 60)}:{int(video_info['duration'] % 60):02d}"
            size_mb = video_info['size'] / (1024 * 1024)
            
            info_text = f"""
📹 *Video Information*

🎬 *Duration:* {duration_str}
📦 *Size:* {size_mb:.1f} MB
📐 *Resolution:* {video_info.get('width', 'N/A')}×{video_info.get('height', 'N/A')}
🎞️ *FPS:* {video_info.get('fps', 'N/A'):.1f}
🎵 *Audio:* {video_info.get('audio_codec', 'N/A')}

Choose your preferred resolution(s):
            """
            
            await status_msg.edit_text(
                info_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error handling video: {e}")
            await update.message.reply_text(f"❌ Error processing video: {str(e)}")
    
    def _estimate_file_size(self, video_info: dict, resolution: str) -> str:
        """Estimate output file size"""
        try:
            original_size = video_info.get('size', 0)
            original_height = video_info.get('height', 720)
            target_height = VideoProcessor.RESOLUTIONS[resolution]['height']
            
            # Rough estimation based on resolution ratio
            ratio = (target_height / original_height) ** 2
            estimated_size = original_size * ratio * 0.6  # Account for better compression
            
            if estimated_size < 1024 * 1024:
                return f"{estimated_size / 1024:.0f}KB"
            else:
                return f"{estimated_size / (1024 * 1024):.1f}MB"
        except:
            return "~MB"
    
    async def handle_conversion(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced conversion handler"""
        query = update.callback_query
        await query.answer()
        
        try:
            # Parse callback data
            parts = query.data.split('_')
            conversion_type = parts[1]
            message_id = parts[2]
            timestamp = parts[3]
            
            input_path = None
            for file_path in self.temp_dir.glob(f"input_{message_id}_*"):
                input_path = file_path
                break
            
            if not input_path or not input_path.exists():
                await query.edit_message_text("❌ Original video not found. Please send the video again.")
                return
            
            # Handle different conversion types
            if conversion_type == "all":
                await self._convert_all_resolutions(query, input_path, message_id, timestamp)
            elif conversion_type == "mobile":
                await self._convert_mobile_pack(query, input_path, message_id, timestamp)
            else:
                await self._convert_single_resolution(query, input_path, conversion_type, message_id, timestamp)
                
        except Exception as e:
            logger.error(f"Error in conversion: {e}")
            await query.edit_message_text(f"❌ Conversion failed: {str(e)}")
    
    async def _convert_single_resolution(self, query, input_path, resolution, message_id, timestamp):
        """Convert to single resolution with progress"""
        await query.edit_text(
            f"🔄 *Converting to {resolution.upper()}...*\n"
            "⏳ This may take a few minutes",
            parse_mode=ParseMode.MARKDOWN
        )
        
        output_path = self.temp_dir / f"output_{resolution}_{message_id}_{timestamp}.mp4"
        
        success = await VideoProcessor.process_video(
            str(input_path), str(output_path), resolution
        )
        
        if success:
            await self._send_processed_video(query, output_path, resolution)
            # Cleanup
            self._cleanup_files(message_id, timestamp)
        else:
            await query.edit_text(f"❌ Failed to convert to {resolution}")
    
    async def _convert_mobile_pack(self, query, input_path, message_id, timestamp):
        """Convert to mobile-optimized resolutions"""
        mobile_resolutions = ['144p', '360p', '480p']
        await query.edit_text(
            "🔄 *Creating Mobile Pack...*\n"
            "📱 Converting to 144p, 360p, 480p",
            parse_mode=ParseMode.MARKDOWN
        )
        
        successful_conversions = []
        
        for i, resolution in enumerate(mobile_resolutions):
            await query.edit_text(
                f"🔄 *Mobile Pack Progress*\n"
                f"📱 Converting to {resolution.upper()} ({i+1}/3)...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            output_path = self.temp_dir / f"output_{resolution}_{message_id}_{timestamp}.mp4"
            success = await VideoProcessor.process_video(
                str(input_path), str(output_path), resolution
            )
            
            if success:
                successful_conversions.append((resolution, output_path))
        
        # Send all successful conversions
        for resolution, output_path in successful_conversions:
            await self._send_processed_video(query, output_path, resolution, is_batch=True)
        
        await query.message.reply_text(f"✅ Mobile Pack completed! Sent {len(successful_conversions)} videos.")
        self._cleanup_files(message_id, timestamp)
    
    async def _convert_all_resolutions(self, query, input_path, message_id, timestamp):
        """Convert to all resolutions"""
        resolutions = list(VideoProcessor.RESOLUTIONS.keys())
        await query.edit_text(
            "🔄 *Converting to All Resolutions...*\n"
            "🎯 This will take several minutes",
            parse_mode=ParseMode.MARKDOWN
        )
        
        successful_conversions = []
        
        for i, resolution in enumerate(resolutions):
            await query.edit_text(
                f"🔄 *Progress: {i+1}/{len(resolutions)}*\n"
                f"📱 Converting to {resolution.upper()}...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            output_path = self.temp_dir / f"output_{resolution}_{message_id}_{timestamp}.mp4"
            success = await VideoProcessor.process_video(
                str(input_path), str(output_path), resolution
            )
            
            if success:
                successful_conversions.append((resolution, output_path))
        
        # Send all conversions
        for resolution, output_path in successful_conversions:
            await self._send_processed_video(query, output_path, resolution, is_batch=True)
        
        await query.message.reply_text(f"🎉 All resolutions completed! Sent {len(successful_conversions)} videos.")
        self._cleanup_files(message_id, timestamp)
    
    async def _send_processed_video(self, query, output_path, resolution, is_batch=False):
        """Enhanced video sending with smart file size handling"""
        try:
            file_size = os.path.getsize(output_path)
            size_mb = file_size / (1024 * 1024)
            
            caption = f"📱 *{resolution.upper()} Video*\n📦 Size: {size_mb:.1f} MB"
            
            # Progress callback for large uploads
            last_progress = [0]
            async def progress_callback(status):
                # Only update every 10% to avoid spam
                if "%" in status:
                    try:
                        current = float(status.split(":")[1].split("%")[0])
                        if current - last_progress[0] >= 10:
                            last_progress[0] = current
                            await query.edit_text(
                                f"📤 *Uploading {resolution.upper()}*\n{status}",
                                parse_mode=ParseMode.MARKDOWN
                            )
                    except:
                        pass
            
            # Try large file upload for files > 50MB
            if file_size > MAX_BOT_API_SIZE:
                uploaded = await self.uploader.upload_large_video(
                    str(output_path), 
                    query.message.chat_id, 
                    caption,
                    progress_callback
                )
                
                if uploaded:
                    if not is_batch:
                        await query.edit_text(f"✅ {resolution.upper()} video sent!")
                    return
            
            # Use regular bot API
            with open(output_path, 'rb') as video_file:
                await query.message.reply_video(
                    video=video_file,
                    caption=caption,
                    supports_streaming=True,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            if not is_batch:
                await query.edit_text(f"✅ {resolution.upper()} video sent!")
                
        except Exception as e:
            logger.error(f"Error sending video: {e}")
            if not is_batch:
                await query.edit_text(f"❌ Failed to send {resolution} video")
    
    def _cleanup_files(self, message_id, timestamp):
        """Clean up temporary files"""
        try:
            patterns = [f"*_{message_id}_*", f"*_{message_id}_{timestamp}*"]
            for pattern in patterns:
                for file_path in self.temp_dir.glob(pattern):
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

async def main():
    """Main function optimized for Railway"""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable not set!")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Create handler instance
    bot_handler = VideoBotHandler()
    
    # Add handlers
    application.add_handler(CommandHandler("start", bot_handler.start))
    application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, bot_handler.handle_video))
    application.add_handler(CallbackQueryHandler(bot_handler.handle_conversion))
    
    # Railway-specific: Use webhook if PORT is available, otherwise polling
    if PORT and os.getenv('RAILWAY_ENVIRONMENT'):
        # Webhook mode for Railway
        webhook_url = f"https://{os.getenv('RAILWAY_STATIC_URL', 'localhost')}"
        logger.info(f"Starting webhook on port {PORT}")
        
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=f"{webhook_url}/{BOT_TOKEN}",
            url_path=BOT_TOKEN
        )
    else:
        # Polling mode for development
        logger.info("Starting polling mode")
        application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
