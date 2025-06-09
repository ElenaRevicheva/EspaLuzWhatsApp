from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
import json
import time
import re
from datetime import datetime
from gtts import gTTS
from requests.auth import HTTPBasicAuth

# App setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# User sessions for WhatsApp (same structure as Telegram bot)
user_sessions = {}

# Family member profiles (copied from main.py)
FAMILY_MEMBERS = {
    "alisa": {
        "role": "child",
        "age": 4,
        "learning_level": "beginner",
        "interests": ["animals", "colors", "games", "songs"],
        "tone": "playful",
        "language_balance": {"spanish": 0.6, "english": 0.4},
    },
    "marina": {
        "role": "elder", 
        "age": 65,
        "learning_level": "beginner",
        "interests": ["cooking", "culture", "daily life", "health"],
        "tone": "patient",
        "language_balance": {"spanish": 0.7, "english": 0.3},
    },
    "elena": {
        "role": "parent",
        "age": 39,
        "learning_level": "intermediate", 
        "interests": ["work", "travel", "parenting", "culture"],
        "tone": "conversational",
        "language_balance": {"spanish": 0.5, "english": 0.5},
    }
}

def create_initial_session(user_id, user_phone, message_text=""):
    """Create initial session for WhatsApp user (adapted from main.py)"""
    # Default to Elena profile for new users
    family_member = "elena"
    member_info = FAMILY_MEMBERS[family_member]
    
    return {
        "messages": [],
        "context": {
            "user": {
                "id": str(user_id),
                "phone": user_phone,
                "preferences": {
                    "family_role": family_member,
                    "age": member_info["age"],
                    "learning_level": member_info["learning_level"],
                    "interests": member_info["interests"],
                    "tone_preference": member_info["tone"],
                    "primary_language": "russian",
                    "target_languages": ["spanish", "english"],
                }
            },
            "emotional_state": {
                "current_emotion": "neutral",
                "emotion_confidence": 1.0,
                "last_emotions": []
            },
            "conversation": {
                "start_time": datetime.now().isoformat(),
                "message_count": 0,
                "recent_topics": [],
                "language_balance": member_info["language_balance"]
            },
            "learning": {
                "total_sessions": 1,
                "progress": {
                    "vocabulary": {
                        "spanish": {"learned": [], "needs_review": []},
                        "english": {"learned": [], "needs_review": []}
                    },
                    "grammar": {
                        "spanish": {"learned": [], "needs_review": []},
                        "english": {"learned": [], "needs_review": []}
                    }
                }
            }
        }
    }

def translate_to_es_en(text):
    """Translate text to Spanish and English using OpenAI"""
    try:
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
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

def ask_claude_standalone(session, user_message, translated_input):
    """Ask Claude for response (standalone version)"""
    try:
        # Get family member info
        family_role = session["context"]["user"]["preferences"]["family_role"]
        member_info = FAMILY_MEMBERS[family_role]
        
        # Create system prompt
        system_content = f"""You are Espaluz, a bilingual emotionally intelligent AI language tutor for a Russian expat family in Panama.

You're speaking with a user who has the profile: {family_role} (age {member_info['age']}, {member_info['learning_level']} level).

Your answer should have TWO PARTS:

1️⃣ A full, thoughtful bilingual response (using both Spanish and English):
   - Respond naturally to the message
   - Be emotionally aware, friendly, and motivating  
   - Include relevant Spanish learning or cultural context from Panama
   - Use vocabulary appropriate for the user's level

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

        # Prepare messages  
        messages = session["messages"][-10:]  # Last 10 messages
        
        # Add current message
        user_content = user_message
        if translated_input:
            user_content += f"\n\n[TRANSLATION]\n{translated_input}"
            
        messages.append({"role": "user", "content": user_content})
        
        # Call Claude API
        headers = {
            "x-api-key": os.getenv("CLAUDE_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        request_data = {
            "model": "claude-sonnet-4-20250514",
            "messages": messages,
            "system": system_content,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        res = requests.post("https://api.anthropic.com/v1/messages",
                           headers=headers, json=request_data)
        
        if res.status_code == 200:
            result = res.json()
            full_reply = result["content"][0]["text"]
            
            # Extract video script
            video_script = extract_video_script(full_reply)
            
            return full_reply.strip(), video_script.strip()
        else:
            logging.error(f"Claude API error: {res.status_code} - {res.text}")
            raise Exception(f"Claude API failed: {res.status_code}")
            
    except Exception as e:
        logging.error(f"Claude error: {e}")
        # Fallback to OpenAI
        return fallback_to_openai(user_message)

def fallback_to_openai(user_message):
    """Fallback to OpenAI if Claude fails"""
    try:
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": """You are Espaluz, a bilingual Spanish-English tutor for Russian expats in Panama. 

Respond with:
1) A helpful bilingual response
2) Then add: [VIDEO SCRIPT START] Short Spanish and English phrases [VIDEO SCRIPT END]"""},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 800
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        
        if res.status_code == 200:
            full_reply = res.json()["choices"][0]["message"]["content"]
            video_script = extract_video_script(full_reply)
            return full_reply, video_script
        else:
            return ("Lo siento, hay un problema técnico. Sorry, there's a technical issue.", 
                   "Lo siento. Sorry about that.")
    except Exception as e:
        logging.error(f"OpenAI fallback error: {e}")
        return ("Lo siento, hay un problema técnico. Sorry, there's a technical issue.", 
               "Lo siento. Sorry about that.")

def extract_video_script(full_response):
    """Extract video script from response"""
    if "[VIDEO SCRIPT START]" in full_response and "[VIDEO SCRIPT END]" in full_response:
        start_idx = full_response.find("[VIDEO SCRIPT START]") + len("[VIDEO SCRIPT START]")
        end_idx = full_response.find("[VIDEO SCRIPT END]", start_idx)
        if start_idx < end_idx:
            return full_response[start_idx:end_idx].strip()
    
    # Fallback
    return "¡Gracias por practicar conmigo! Thank you for practicing with me!"

def generate_tts_audio(text, output_file):
    """Generate TTS audio file"""
    try:
        # Clean text for speech
        clean_text = re.sub(r'[^\w\s,.?!:;\'-]', '', text)
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
            
        tts = gTTS(text=clean_text, lang="es", slow=False)
        tts.save(output_file)
        return os.path.exists(output_file)
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return False

# Webhook verification (for both Meta and Twilio)
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token") 
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "espaluz123":
        logging.info("✅ Webhook verified")
        return challenge, 200
    else:
        logging.warning("❌ Webhook verification failed")
        return "Verification failed", 403

# Handle incoming WhatsApp messages
@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        content_type = request.headers.get('Content-Type', '')
        logging.info(f"🌐 Incoming webhook, Content-Type: {content_type}")
        
        # Handle Twilio format (form data)
        if 'application/x-www-form-urlencoded' in content_type:
            user_id = request.form.get('From', '').replace('whatsapp:', '')
            user_text = request.form.get('Body', '')
            
            if not user_id or not user_text:
                return "ok", 200
                
            logging.info(f"📩 Message from {user_id}: {user_text}")
            
            # Process message
            response_text = process_whatsapp_message(user_id, user_text)
            
            # Send response
            send_whatsapp_message_twilio(user_id, response_text)
            return "ok", 200
            
        else:
            return jsonify({"error": "Unsupported content type"}), 400
            
    except Exception as e:
        logging.exception("❌ Error handling message")
        return jsonify({"error": str(e)}), 500

def process_whatsapp_message(user_id, user_text):
    """Process WhatsApp message and return response"""
    try:
        # Initialize session if needed
        if user_id not in user_sessions:
            user_sessions[user_id] = create_initial_session(user_id, user_id, user_text)
            logging.info(f"Created new session for {user_id}")

        session = user_sessions[user_id]
        session["context"]["conversation"]["message_count"] += 1
        
        # Get translation
        translated = translate_to_es_en(user_text)
        logging.info(f"Translation: {translated[:100]}...")
        
        # Update message history
        session["messages"].append({"role": "user", "content": user_text})
        
        # Get Claude response
        full_reply, video_script = ask_claude_standalone(session, user_text, translated)
        logging.info(f"Claude response length: {len(full_reply)}")
        
        # Update session
        session["messages"].append({"role": "assistant", "content": full_reply})
        
        # For now, return the full text response
        # TODO: Add voice/video generation later
        return f"📝 Traducción:\n{translated}\n\n🤖 Espaluz:\n{full_reply}"
        
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return "Lo siento, hubo un error. Sorry, there was an error."

def send_whatsapp_message_twilio(to_number, message_text):
    """Send WhatsApp message via Twilio"""
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        
        data = {
            'From': 'whatsapp:+14155238886',
            'To': f'whatsapp:{to_number}',
            'Body': message_text[:1600]  # Twilio character limit
        }
        
        response = requests.post(url, data=data, auth=HTTPBasicAuth(account_sid, auth_token))
        
        logging.info(f"📤 Sent to {to_number}, status: {response.status_code}")
        if response.status_code != 201:
            logging.error(f"Twilio error: {response.text}")
            
    except Exception as e:
        logging.error(f"❌ Error sending message: {e}")

# Health check
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Espaluz WhatsApp Bridge - Standalone",
        "version": "3.0-standalone"
    })

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🚀 Starting Standalone Espaluz WhatsApp Bridge on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)