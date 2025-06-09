from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
import re
from datetime import datetime
from gtts import gTTS
import subprocess

# App setup
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
    return "Gracias por practicar conmigo. Thank you for practicing with me."

def generate_tts_audio(text, filename):
    try:
        cleaned = re.sub(r'[^\w\s,.?!:;\'-]', '', text)
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        tts = gTTS(text=cleaned, lang="es")
        tts.save(filename)
        return os.path.exists(filename)
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return False

def create_video_with_audio(script_text, output_path):
    try:
        tts_path = "/tmp/tmp_audio.mp3"
        if not generate_tts_audio(script_text, tts_path):
            return False
        base_video = "looped_video.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", base_video, "-i", tts_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        logging.error(f"FFmpeg error: {e}")
        return False

def ask_claude(user_text):
    try:
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        prompt = f"""
You are Espaluz, a bilingual emotional tutor for expat families.
Respond in 2 parts:
1) Full bilingual reply (Spanish + English)
2) Short [VIDEO SCRIPT START] ... [END] block for spoken summary
Today is {datetime.now().strftime('%Y-%m-%d')}.
"""
        messages = [{"role": "user", "content": user_text}]
        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": messages,
            "system": prompt,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        reply = r.json()["content"][0]["text"]
        return reply.strip(), extract_video_script(reply.strip())
    except Exception as e:
        logging.error(f"Claude error: {e}")
        return fallback_openai(user_text)

def fallback_openai(user_text):
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        messages = [
            {"role": "system", "content": "Bilingual tutor. Reply fully then add [VIDEO SCRIPT START]..."},
            {"role": "user", "content": user_text}
        ]
        data = {
            "model": "gpt-4",
            "messages": messages,
            "max_tokens": 800
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        full = r.json()["choices"][0]["message"]["content"]
        return full, extract_video_script(full)
    except:
        return "Lo siento. Sorry.", "Lo siento. Sorry."

def send_whatsapp_message(to, text):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        'From': TWILIO_NUMBER,
        'To': f"whatsapp:{to}",
        'Body': text[:1600]
    }
    res = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    logging.info(f"📤 Sent message to {to} status: {res.status_code}")

def send_whatsapp_media(to, file_path, media_type="audio"):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    media_url = upload_media_to_tmp_host(file_path)  # Placeholder
    data = {
        'From': TWILIO_NUMBER,
        'To': f"whatsapp:{to}",
        'Body': "",
        'MediaUrl': media_url
    }
    res = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    logging.info(f"📤 Sent {media_type} to {to} status: {res.status_code}")

def upload_media_to_tmp_host(local_path):
    # Temporarily use Dropbox or Fleek hosted URLs
    # For now, simulate with dummy hosted file
    if "audio" in local_path:
        return "https://aideazz.xyz/tmp/reply_audio.mp3"
    else:
        return "https://aideazz.xyz/tmp/espaluz_video.mp4"

@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        user_id = request.form.get("From", "").replace("whatsapp:", "")
        user_text = request.form.get("Body", "")
        media_count = int(request.form.get("NumMedia", 0))

        if media_count > 0:
            send_whatsapp_message(user_id, "📸 I received your image! OCR is coming soon.")
            return "ok", 200

        if not user_text:
            return "ok", 200

        logging.info(f"📩 Message from {user_id}: {user_text}")

        full_reply, video_script = ask_claude(user_text)

        send_whatsapp_message(user_id, f"🤖 Espaluz:\n{full_reply}")

        audio_path = "/tmp/reply_audio.mp3"
        if generate_tts_audio(full_reply, audio_path):
            send_whatsapp_media(user_id, audio_path, media_type="audio")

        video_path = "/tmp/espaluz_video.mp4"
        if create_video_with_audio(video_script, video_path):
            send_whatsapp_media(user_id, video_path, media_type="video")

        return "ok", 200

    except Exception as e:
        logging.exception("❌ Error in webhook")
        return "ok", 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "bot": "Espaluz WhatsApp Sandbox",
        "version": "v1.0"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🚀 Starting on port {port}")
    app.run(host="0.0.0.0", port=port)
