from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
import re
import subprocess
from datetime import datetime
from gtts import gTTS
from requests.auth import HTTPBasicAuth

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

TWILIO_NUMBER = "whatsapp:+14155238886"  # Twilio sandbox number

# Claude + OpenAI API setup
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

def extract_video_script(full_response):
    if "[VIDEO SCRIPT START]" in full_response and "[VIDEO SCRIPT END]" in full_response:
        start_idx = full_response.find("[VIDEO SCRIPT START]") + len("[VIDEO SCRIPT START]")
        end_idx = full_response.find("[VIDEO SCRIPT END]", start_idx)
        return full_response[start_idx:end_idx].strip()
    return "¡Gracias por practicar conmigo! Thank you for practicing with me!"

def generate_tts_audio(text, filename):
    try:
        # Clean text for speech
        cleaned = re.sub(r'[^\w\s,.?!:;\'-]', '', text)
        # Limit length to avoid very long audio
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
            'Body': text[:1500]  # Ensure under Twilio limit
        }
        res = requests.post(url, data=data, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        logging.info(f"📤 Sent text to {to} status: {res.status_code}")
        
        if res.status_code != 201:
            logging.error(f"Text message error: {res.text}")
            
    except Exception as e:
        logging.error(f"Error sending text message: {e}")

def send_whatsapp_audio(to, audio_file_path):
    """Send audio file via Twilio Media API"""
    try:
        # Upload to Twilio Media first
        upload_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Media.json"
        
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            upload_response = requests.post(
                upload_url, 
                files=files, 
                auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            )
        
        if upload_response.status_code == 201:
            media_url = upload_response.json()['uri']
            media_url = f"https://api.twilio.com{media_url}"
            
            # Send message with media
            message_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            data = {
                'From': TWILIO_NUMBER,
                'To': f"whatsapp:{to}",
                'MediaUrl': media_url
            }
            
            response = requests.post(
                message_url, 
                data=data, 
                auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            )
            logging.info(f"🎧 Sent audio to {to} status: {response.status_code}")
            
        else:
            logging.error(f"Audio upload failed: {upload_response.text}")
            
    except Exception as e:
        logging.error(f"Error sending audio: {e}")

def send_whatsapp_video(to, video_file_path):
    """Send video file via Twilio Media API"""
    try:
        # Upload to Twilio Media first
        upload_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Media.json"
        
        with open(video_file_path, 'rb') as f:
            files = {'file': f}
            upload_response = requests.post(
                upload_url, 
                files=files, 
                auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            )
        
        if upload_response.status_code == 201:
            media_url = upload_response.json()['uri']
            media_url = f"https://api.twilio.com{media_url}"
            
            # Send message with media
            message_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            data = {
                'From': TWILIO_NUMBER,
                'To': f"whatsapp:{to}",
                'MediaUrl': media_url
            }
            
            response = requests.post(
                message_url, 
                data=data, 
                auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            )
            logging.info(f"🎥 Sent video to {to} status: {response.status_code}")
            
        else:
            logging.error(f"Video upload failed: {upload_response.text}")
            
    except Exception as e:
        logging.error(f"Error sending video: {e}")

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Webhook verification for Meta/Twilio"""
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

        # Handle media messages (images)
        if media_count > 0:
            send_whatsapp_message(user_id, "📸 ¡Recibí tu imagen! Pronto tendré OCR. / I received your image! OCR coming soon.")
            return "ok", 200

        # Handle text messages
        if not user_text:
            return "ok", 200

        logging.info(f"📩 Message from {user_id}: {user_text}")

        # Get Claude response
        full_reply, video_script = ask_claude(user_text)

        # Send text response
        send_whatsapp_message(user_id, f"🤖 Espaluz:\n{full_reply}")

        # Generate and send audio
        audio_path = "/tmp/reply_audio.mp3"
        if generate_tts_audio(full_reply, audio_path):
            send_whatsapp_audio(user_id, audio_path)
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Generate and send video
        video_path = "/tmp/espaluz_video.mp4"
        if create_simple_video(video_script, video_path):
            send_whatsapp_video(user_id, video_path)
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)

        return "ok", 200

    except Exception as e:
        logging.exception("❌ Error in webhook")
        return "ok", 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "bot": "Espaluz WhatsApp Complete Multimedia",
        "version": "v2.0-fixed",
        "features": ["text", "voice", "video", "claude_ai"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🚀 Starting Complete Espaluz on port {port}")
    app.run(host="0.0.0.0", port=port)