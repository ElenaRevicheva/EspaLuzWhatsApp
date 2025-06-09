from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
import requests
import re
import subprocess
import base64
import io
import threading
import time
from datetime import datetime
from gtts import gTTS
from requests.auth import HTTPBasicAuth
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

TWILIO_NUMBER = "whatsapp:+14155238886"

# API setup
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Create media directory
os.makedirs("/tmp/media", exist_ok=True)

def clean_text_for_speech(text: str) -> str:
    """Remove punctuation marks and formatting for natural speech"""
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)  # Remove asterisks
    text = re.sub(r'_+([^_]+)_+', r'\1', text)    # Remove underscores
    text = re.sub(r'`+([^`]+)`+', r'\1', text)    # Remove backticks
    text = re.sub(r'"([^"]+)"', r'\1', text)      # Remove quotes
    text = re.sub(r"'([^']+)'", r'\1', text)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # Remove list numbers
    text = re.sub(r'\n\d+\.\s*', '\n', text)
    text = re.sub(r'[0-9]️⃣', '', text)            # Remove emoji numbers
    text = re.sub(r'[#@\[\](){}<>]', '', text)    # Remove symbols
    text = re.sub(r'\s+', ' ', text)              # Clean whitespace
    return text.strip()

def extract_video_script(full_response):
    if "[VIDEO SCRIPT START]" in full_response and "[VIDEO SCRIPT END]" in full_response:
        start_idx = full_response.find("[VIDEO SCRIPT START]") + len("[VIDEO SCRIPT START]")
        end_idx = full_response.find("[VIDEO SCRIPT END]", start_idx)
        return full_response[start_idx:end_idx].strip()
    return "¡Gracias por practicar conmigo! Thank you for practicing with me!"

def generate_tts_audio(text, filename):
    try:
        cleaned = clean_text_for_speech(text)
        if len(cleaned) > 800:
            cleaned = cleaned[:800] + "..."
        tts = gTTS(text=cleaned, lang="es", slow=False)
        tts.save(filename)
        return os.path.exists(filename)
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return False

def create_simple_video(script_text, output_path):
    """Create a simple video with text and audio"""
    try:
        # Generate TTS audio first
        tts_path = "/tmp/tmp_audio.mp3"
        if not generate_tts_audio(script_text, tts_path):
            return False
        
        # Create simple colored video with text overlay
        clean_text = script_text.replace("'", "").replace('"', '').replace('\\', '')[:50]
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=640x480:d=10",
            "-i", tts_path,
            "-vf", f"drawtext=text='{clean_text}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp audio
        if os.path.exists(tts_path):
            os.remove(tts_path)
            
        return result.returncode == 0 and os.path.exists(output_path)
        
    except Exception as e:
        logging.error(f"Video creation error: {e}")
        return False

def transcribe_audio_with_openai(audio_file_path):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        with open(audio_file_path, 'rb') as f:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": f},
                data={"model": "whisper-1"}
            )
        
        if response.status_code == 200:
            transcription = response.json().get("text", "")
            logging.info(f"🎤 Transcribed: {transcription}")
            return transcription
        else:
            logging.error(f"Transcription failed: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

def translate_to_es_en(text):
    """Translate text to Spanish and English using OpenAI"""
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Translate this message into both Spanish and English:\n\n{text}"}],
            "max_tokens": 500
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"Translation: {text} (translation service unavailable)"
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return f"Translation: {text} (error in translation)"

def convert_audio_format(input_path, output_path):
    """Convert audio file to format suitable for Whisper (MP3/WAV)"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-acodec", "mp3",
            "-ar", "16000",  # 16kHz sample rate for Whisper
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            logging.info(f"🔄 Audio converted: {output_path}")
            return True
        else:
            logging.error(f"Audio conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Audio conversion error: {e}")
        return False

def process_whatsapp_voice_async(user_id, media_url):
    """Process voice message asynchronously"""
    def process():
        try:
            logging.info(f"🎤 Starting voice processing for {user_id}")
            
            # Send initial processing message
            send_whatsapp_message(user_id, "🎤 Procesando mensaje de voz... / Processing voice message...")
            
            # Download audio file
            logging.info(f"🔗 Downloading audio from: {media_url}")
            response = requests.get(media_url, timeout=30)
            if response.status_code != 200:
                logging.error(f"Audio download failed: {response.status_code}")
                send_whatsapp_message(user_id, "❌ No pude descargar el audio. Could not download audio.")
                return
            
            logging.info(f"✅ Audio downloaded, size: {len(response.content)} bytes")
            
            # Save downloaded audio
            timestamp = int(time.time())
            original_audio = f"/tmp/voice_original_{timestamp}.ogg"
            converted_audio = f"/tmp/voice_converted_{timestamp}.mp3"
            
            with open(original_audio, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"💾 Audio saved to: {original_audio}")
            
            # Convert audio format for Whisper
            logging.info("🔄 Converting audio format...")
            if not convert_audio_format(original_audio, converted_audio):
                send_whatsapp_message(user_id, "❌ Error convirtiendo audio. Error converting audio.")
                return
            
            logging.info(f"✅ Audio converted to: {converted_audio}")
            
            # Transcribe audio
            logging.info("🗣️ Transcribing with Whisper...")
            transcription = transcribe_audio_with_openai(converted_audio)
            
            if not transcription or len(transcription.strip()) == 0:
                logging.error("Transcription failed or empty")
                send_whatsapp_message(user_id, "❌ No pude transcribir el audio. Could not transcribe audio.")
                return
            
            logging.info(f"✅ Transcription successful: {transcription}")
            
            # Send transcription
            send_whatsapp_message(user_id, f"🗣️ Transcripción / Transcription:\n{transcription}")
            
            # Get translation
            logging.info("🌐 Translating...")
            translation = translate_to_es_en(transcription)
            send_whatsapp_message(user_id, f"📝 Traducción / Translation:\n{translation}")
            
            # Process transcription as text message (get Claude response + multimedia)
            logging.info("🤖 Processing with Claude...")
            process_transcribed_message(user_id, transcription)
            
            # Clean up temp files
            for file_path in [original_audio, converted_audio]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            logging.info("🧹 Cleanup completed")
                    
        except Exception as e:
            logging.exception(f"❌ Error processing voice message: {e}")
            send_whatsapp_message(user_id, "❌ Error procesando mensaje de voz. Error processing voice message.")
    
    # Run in background thread
    thread = threading.Thread(target=process, daemon=True)
    thread.start()
    logging.info(f"🚀 Voice processing thread started for {user_id}")

def process_transcribed_message(user_id, transcription):
    """Process transcribed text and generate multimedia response"""
    try:
        # Get Claude response
        full_reply, video_script = ask_claude(transcription)
        
        # Send Claude response
        send_whatsapp_message(user_id, f"🤖 Espaluz:\n{full_reply}")
        
        # Generate and send audio response
        audio_path = f"/tmp/reply_audio_{int(time.time())}.mp3"
        if generate_tts_audio(full_reply, audio_path):
            send_whatsapp_media(user_id, audio_path, "audio")
        
        # Generate and send video response
        video_path = f"/tmp/espaluz_video_{int(time.time())}.mp4"
        if create_simple_video(video_script, video_path):
            send_whatsapp_media(user_id, video_path, "video")
            
    except Exception as e:
        logging.error(f"Error processing transcribed message: {e}")
    """Process image using GPT-4 Vision - Enhanced OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert at extracting and translating text from images. 
                    Extract ALL visible text, preserving formatting and structure.
                    Then provide translations to both Spanish and English with cultural context for Russian expats in Panama."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ALL text from this image. Preserve structure and formatting. Then translate to Spanish and English with cultural insights for Russian expats in Panama."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                        }
                    ]
                }
            ],
            "max_tokens": 3000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            extracted_text = result["choices"][0]["message"]["content"]
            word_count = len(extracted_text.split())
            return f"📄 Extracted {word_count} words:\n\n{extracted_text}"
        else:
            return "❌ Could not process the image."

    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return "❌ Error processing the image."

def ask_claude(user_text):
    try:
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        system_prompt = f"""You are Espaluz, a bilingual emotionally intelligent AI language tutor for a Russian expat family in Panama.

Your answer should have TWO PARTS:

1️⃣ A full, thoughtful bilingual response (using both Spanish and English):
   - Respond naturally to the message
   - Be emotionally aware, friendly, and motivating  
   - Include relevant Spanish learning or cultural context from Panama
   - Keep it concise but helpful (under 800 characters)

2️⃣ A second short block inside [VIDEO SCRIPT START] ... [VIDEO SCRIPT END] for video:
   - Must be 2 to 4 concise sentences MAX
   - Use both Spanish and English
   - Tone: warm, clear, and simple for spoken delivery
   
Example:
[VIDEO SCRIPT START]
¡Hola! Hoy es un gran día para aprender español.
Hello! Today is a great day to learn Spanish.
[VIDEO SCRIPT END]

Today is {datetime.now().strftime('%Y-%m-%d')}."""

        messages = [{"role": "user", "content": user_text}]
        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": messages,
            "system": system_prompt,
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        
        if r.status_code == 200:
            reply = r.json()["content"][0]["text"]
            return reply.strip(), extract_video_script(reply.strip())
        else:
            logging.error(f"Claude API error: {r.status_code} - {r.text}")
            return fallback_openai(user_text)
            
    except Exception as e:
        logging.error(f"Claude error: {e}")
        return fallback_openai(user_text)

def fallback_openai(user_text):
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        messages = [
            {"role": "system", "content": """You are Espaluz, a bilingual Spanish-English tutor. 
            
Respond with:
1) A helpful bilingual response (keep under 800 characters)
2) Then add: [VIDEO SCRIPT START] Short Spanish and English phrases [VIDEO SCRIPT END]"""},
            {"role": "user", "content": user_text}
        ]
        data = {
            "model": "gpt-4",
            "messages": messages,
            "max_tokens": 600
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        
        if r.status_code == 200:
            full = r.json()["choices"][0]["message"]["content"]
            return full, extract_video_script(full)
        else:
            return "Lo siento, hay un problema técnico. Sorry, there's a technical issue.", "Lo siento. Sorry."
    except Exception as e:
        logging.error(f"OpenAI fallback error: {e}")
        return "Lo siento, hay un problema técnico. Sorry, there's a technical issue.", "Lo siento. Sorry."

def send_whatsapp_message(to, text):
    """Send text message via Twilio"""
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            'From': TWILIO_NUMBER,
            'To': f"whatsapp:{to}",
            'Body': text[:1500]
        }
        res = requests.post(url, data=data, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        logging.info(f"📤 Sent text to {to} status: {res.status_code}")
        
        if res.status_code != 201:
            logging.error(f"Text message error: {res.text}")
            
    except Exception as e:
        logging.error(f"Error sending text message: {e}")

def send_whatsapp_media(to, file_path, media_type="audio"):
    """Send media file via hosted URL"""
    try:
        # Create unique filename
        timestamp = int(time.time())
        filename = f"{media_type}_{timestamp}_{os.path.basename(file_path)}"
        
        # Copy file to accessible location
        hosted_path = f"/tmp/media/{filename}"
        subprocess.run(["cp", file_path, hosted_path], check=True)
        
        # Create public URL (Railway serves files from /tmp/media/)
        railway_app_url = os.getenv("RAILWAY_PUBLIC_DOMAIN", "espaluzwhatsapp-production.up.railway.app")
        media_url = f"https://{railway_app_url}/media/{filename}"
        
        # Send message with media URL
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            'From': TWILIO_NUMBER,
            'To': f"whatsapp:{to}",
            'Body': f"🎧 Audio message" if media_type == "audio" else "🎥 Video message",
            'MediaUrl': media_url
        }
        
        res = requests.post(url, data=data, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        logging.info(f"📤 Sent {media_type} to {to} status: {res.status_code} - URL: {media_url}")
        
        if res.status_code != 201:
            logging.error(f"{media_type.title()} send failed: {res.text}")
        
        # Clean up original file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        logging.error(f"Error sending {media_type}: {e}")

def process_whatsapp_message_async(user_id, user_text):
    """Process message and send multimedia response asynchronously"""
    def process():
        try:
            logging.info(f"📩 Processing message from {user_id}: {user_text}")
            
            # Get Claude response
            full_reply, video_script = ask_claude(user_text)
            
            # Send text response
            send_whatsapp_message(user_id, f"🤖 Espaluz:\n{full_reply}")
            
            # Generate and send audio
            audio_path = f"/tmp/reply_audio_{int(time.time())}.mp3"
            if generate_tts_audio(full_reply, audio_path):
                send_whatsapp_media(user_id, audio_path, "audio")
            
            # Generate and send video
            video_path = f"/tmp/espaluz_video_{int(time.time())}.mp4"
            if create_simple_video(video_script, video_path):
                send_whatsapp_media(user_id, video_path, "video")
                
        except Exception as e:
            logging.error(f"Error in async processing: {e}")
            send_whatsapp_message(user_id, "Lo siento, hubo un error. Sorry, there was an error.")
    
    # Run in background thread
    thread = threading.Thread(target=process, daemon=True)
    thread.start()

def process_whatsapp_image_async(user_id, media_url):
    """Process image message asynchronously"""
    def process():
        try:
            # Download image
            response = requests.get(media_url)
            if response.status_code == 200:
                send_whatsapp_message(user_id, "🔍 Procesando imagen... / Processing image...")
                
                # Process with GPT-4 Vision
                result = process_image_with_gpt4_vision(response.content)
                
                # Send result in chunks if too long
                max_length = 1500
                if len(result) > max_length:
                    chunks = [result[i:i+max_length] for i in range(0, len(result), max_length)]
                    for i, chunk in enumerate(chunks, 1):
                        chunk_msg = f"📷 Parte {i}/{len(chunks)}:\n{chunk}"
                        send_whatsapp_message(user_id, chunk_msg)
                        time.sleep(1)
                else:
                    send_whatsapp_message(user_id, f"📷 Resultado:\n{result}")
            else:
                send_whatsapp_message(user_id, "❌ No pude descargar la imagen. Could not download the image.")
                
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            send_whatsapp_message(user_id, "❌ Error procesando la imagen. Error processing image.")
    
    # Run in background thread
    thread = threading.Thread(target=process, daemon=True)
    thread.start()

# Static file serving for media
@app.route("/media/<filename>")
def serve_media(filename):
    """Serve media files"""
    try:
        file_path = f"/tmp/media/{filename}"
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
    except Exception as e:
        logging.error(f"Error serving media: {e}")
        return "Error serving file", 500

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Webhook verification"""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "espaluz123":
        logging.info("✅ Webhook verified")
        return challenge, 200
    else:
        logging.warning("❌ Webhook verification failed")
        return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        user_id = request.form.get("From", "").replace("whatsapp:", "")
        user_text = request.form.get("Body", "")
        media_count = int(request.form.get("NumMedia", 0))

        if not user_id:
            return "ok", 200

        # Handle media messages (images, voice, etc.)
        if media_count > 0:
            media_url = request.form.get("MediaUrl0", "")
            media_type = request.form.get("MediaContentType0", "")
            
            logging.info(f"📎 Media received - Type: {media_type}, URL: {media_url}")
            
            if 'image' in media_type:
                # Process image with OCR
                process_whatsapp_image_async(user_id, media_url)
            elif 'audio' in media_type or 'ogg' in media_type:
                # Process voice message
                process_whatsapp_voice_async(user_id, media_url)
            else:
                send_whatsapp_message(user_id, f"📄 Recibí tu archivo ({media_type}). I received your file.")
            
            return "ok", 200

        # Handle text messages
        if not user_text:
            return "ok", 200

        # Process message asynchronously
        process_whatsapp_message_async(user_id, user_text)
        return "ok", 200

    except Exception as e:
        logging.exception("❌ Error in webhook")
        return "ok", 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "bot": "Espaluz WhatsApp Production",
        "version": "v3.0-production",
        "features": ["text", "voice", "video", "image_processing", "voice_transcription", "claude_ai"],
        "endpoints": {
            "webhook": "/webhook",
            "media": "/media/<filename>",
            "health": "/"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🚀 Starting Production Espaluz on port {port}")
    app.run(host="0.0.0.0", port=port)