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

# Enhanced country and cultural context system
COUNTRY_CONTEXTS = {
    "panama": {
        "currency": "USD (Dólar)",
        "transport": ["metro", "metrobús", "taxi", "uber"],
        "landmarks": ["Casco Viejo", "Canal de Panamá", "Cerro Ancón", "Amador Causeway"],
        "food": ["sancocho", "patacones", "ceviche", "arroz con pollo", "tamales"],
        "greetings": ["¿Qué xopá?", "¿Cómo andas?", "¡Épale!"],
        "cultural_tips": "Panamanians are very welcoming to expat families. The country uses USD, making it easy for American families.",
        "climate": "tropical with rainy (May-Nov) and dry seasons (Dec-Apr)",
        "expat_areas": ["Casco Viejo", "San Francisco", "Punta Pacifica", "El Cangrejo"],
        "language_note": "Panamanian Spanish is clear and relatively easy to understand"
    },
    "mexico": {
        "currency": "MXN (Peso Mexicano)",
        "transport": ["metro", "metrobús", "pesero", "uber", "taxi"],
        "landmarks": ["Zócalo", "Chapultepec", "Teotihuacán", "Frida Kahlo Museum"],
        "food": ["tacos", "quesadillas", "mole", "pozole", "chiles rellenos", "tamales"],
        "greetings": ["¿Qué onda?", "¿Cómo estás?", "¡Órale!", "¿Qué tal?"],
        "cultural_tips": "Mexico has huge expat communities, especially in CDMX, Playa del Carmen, and Puerto Vallarta.",
        "climate": "varies greatly - tropical coasts, temperate highlands, desert north",
        "expat_areas": ["Roma Norte", "Condesa", "Polanco", "Playa del Carmen", "San Miguel de Allende"],
        "language_note": "Mexican Spanish has many indigenous influences and regional variations"
    },
    "costa_rica": {
        "currency": "CRC (Colón)",
        "transport": ["autobús", "taxi", "uber", "rental car recommended"],
        "landmarks": ["Volcán Arenal", "Manuel Antonio", "Monteverde", "Tortuguero"],
        "food": ["gallo pinto", "casado", "olla de carne", "tres leches", "chifrijo"],
        "greetings": ["¡Pura vida!", "¿Cómo está?", "¡Mae!", "¿Qué tal?"],
        "cultural_tips": "Costa Rica is extremely expat-friendly with 'Pura Vida' lifestyle. Great for families seeking nature.",
        "climate": "tropical with distinct wet and dry seasons",
        "expat_areas": ["Escazú", "Santa Ana", "Atenas", "Manuel Antonio", "Tamarindo"],
        "language_note": "Costa Rican Spanish is very clear and considered among the easiest to understand"
    },
    "colombia": {
        "currency": "COP (Peso Colombiano)", 
        "transport": ["TransMilenio", "metro", "taxi", "uber"],
        "landmarks": ["Cartagena", "Bogotá Centro", "Medellín Paisa", "Zona Rosa"],
        "food": ["arepa", "bandeja paisa", "ajiaco", "empanadas", "sancocho"],
        "greetings": ["¿Qué más?", "¿Cómo vas?", "¡Bacano!", "¿Todo bien?"],
        "cultural_tips": "Colombia has transformed into a major expat destination, especially Medellín and Bogotá.",
        "climate": "varied - eternal spring in Medellín, cooler in Bogotá, hot on coasts",
        "expat_areas": ["Zona Rosa (Bogotá)", "El Poblado (Medellín)", "Bocagrande (Cartagena)"],
        "language_note": "Colombian Spanish is clear and melodic, varies significantly by region"
    },
    "chile": {
        "currency": "CLP (Peso Chileno)",
        "transport": ["metro", "micro", "colectivo", "uber"],
        "landmarks": ["Cerro San Cristóbal", "Valparaíso", "Atacama Desert", "Torres del Paine"],
        "food": ["empanadas", "asado", "pastel de choclo", "cazuela", "completo"],
        "greetings": ["¿Cómo estai?", "¿Qué tal?", "¡Bacán!", "¿Todo bien?"],
        "cultural_tips": "Chile is very developed and safe, popular with professional expat families.",
        "climate": "Mediterranean in center, desert north, cold south",
        "expat_areas": ["Las Condes", "Providencia", "Ñuñoa", "Valparaíso"],
        "language_note": "Chilean Spanish is fast and uses many unique expressions (chilenismos)"
    },
    "argentina": {
        "currency": "ARS (Peso Argentino)",
        "transport": ["subte", "colectivo", "taxi", "uber"],
        "landmarks": ["Puerto Madero", "Recoleta", "San Telmo", "Tigre Delta"],
        "food": ["asado", "empanadas", "milanesa", "dulce de leche", "mate"],
        "greetings": ["¿Cómo andás?", "¿Todo bien?", "¡Che!", "¿Qué tal?"],
        "cultural_tips": "Argentina has a large expat community, especially Buenos Aires. Very European feel.",
        "climate": "temperate, opposite seasons to Northern Hemisphere",
        "expat_areas": ["Palermo", "Recoleta", "Puerto Madero", "Belgrano"],
        "language_note": "Argentine Spanish has Italian influences, uses 'vos' instead of 'tú'"
    },
    "el_salvador": {
        "currency": "USD (Dólar)",
        "transport": ["autobús", "taxi", "uber"],
        "landmarks": ["Ruta de las Flores", "Suchitoto", "Playa El Tunco", "Volcán de Izalco"],
        "food": ["pupusas", "yuca frita", "curtido", "horchata", "atol de elote"],
        "greetings": ["¿Cómo estás?", "¿Qué tal?", "¡Órale!", "¿Todo bien?"],
        "cultural_tips": "Growing expat destination, uses USD currency, known for surfing and volcanoes.",
        "climate": "tropical with wet and dry seasons",
        "expat_areas": ["San Salvador", "La Libertad", "El Tunco", "Santa Ana"],
        "language_note": "Salvadoran Spanish is clear and influenced by indigenous Nahuatl"
    },
    "spain": {
        "currency": "EUR (Euro)",
        "transport": ["metro", "autobús", "renfe", "ave (high-speed train)"],
        "landmarks": ["Prado", "Sagrada Familia", "Alhambra", "Camino de Santiago"],
        "food": ["paella", "jamón ibérico", "tapas", "gazpacho", "tortilla española"],
        "greetings": ["¿Qué tal?", "¿Cómo va?", "¡Vale!", "¿Todo bien?"],
        "cultural_tips": "Spain offers excellent quality of life, great healthcare, and is very family-friendly.",
        "climate": "Mediterranean coastal, continental interior, varied regions",
        "expat_areas": ["Madrid Centro", "Barcelona Eixample", "Valencia", "Málaga", "Sevilla"],
        "language_note": "Peninsular Spanish uses 'vosotros' and has the distinctive 'theta' sound"
    },
    "peru": {
        "currency": "PEN (Sol)",
        "transport": ["metropolitano", "combi", "taxi", "uber"],
        "landmarks": ["Machu Picchu", "Centro Histórico Lima", "Cusco", "Arequipa"],
        "food": ["ceviche", "lomo saltado", "ají de gallina", "anticuchos", "pisco sour"],
        "greetings": ["¿Cómo estás?", "¿Qué tal?", "¡Causa!", "¿Todo bien?"],
        "cultural_tips": "Peru is becoming popular with expats, especially Lima and Cusco. Rich indigenous culture.",
        "climate": "desert coast, highland mountains, jungle east",
        "expat_areas": ["Miraflores", "San Isidro", "Barranco", "Cusco Centro"],
        "language_note": "Peruvian Spanish is clear, with some Quechua and Aymara influences"
    }
}

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
    """Extract video script from Claude response, ensuring it's clean"""
    if "[VIDEO SCRIPT START]" in full_response and "[VIDEO SCRIPT END]" in full_response:
        start_idx = full_response.find("[VIDEO SCRIPT START]") + len("[VIDEO SCRIPT START]")
        end_idx = full_response.find("[VIDEO SCRIPT END]", start_idx)
        script = full_response[start_idx:end_idx].strip()
        
        # Clean the script and ensure it's reasonable length
        script = clean_text_for_speech(script)
        if len(script) > 200:  # Limit for 30-second video
            script = script[:200] + "..."
        
        return script if script else "¡Hola familia! Hello family! ¡Aprendamos español juntos! Let's learn Spanish together!"
    
    # Fallback if no script found
    return "¡Hola familia! Hello family! ¡Gracias por practicar conmigo! Thank you for practicing with me!"

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

def create_video_with_looped_base(script_text, output_path):
    """Create video using looped_video.mp4 (30 seconds) with VIDEO SCRIPT content"""
    try:
        # Use the looped_video.mp4 as base (30 seconds)
        base_video = "looped_video.mp4"
        
        # Check if base video exists
        if not os.path.exists(base_video):
            logging.error(f"Base video {base_video} not found")
            return False
        
        # Generate TTS audio for the VIDEO SCRIPT content only
        tts_path = "/tmp/video_script_audio.mp3"
        
        # Clean the script text for natural speech
        cleaned_script = clean_text_for_speech(script_text)
        
        logging.info(f"🎬 Creating video with script: {cleaned_script[:100]}...")
        
        # Generate TTS with the video script content
        tts = gTTS(text=cleaned_script, lang="es", slow=False)
        tts.save(tts_path)
        
        if not os.path.exists(tts_path):
            logging.error(f"Failed to create TTS audio: {tts_path}")
            return False
        
        logging.info(f"🔊 Generated TTS for video script: {len(cleaned_script)} characters")
        
        # Combine the 30-second looped video with the script audio
        cmd = [
            "ffmpeg", "-y",
            "-i", base_video,        # Input: looped_video.mp4 (30 seconds)
            "-i", tts_path,          # Input: video script audio
            "-map", "0:v:0",         # Map video from looped_video.mp4
            "-map", "1:a:0",         # Map audio from script TTS
            "-c:v", "copy",          # Copy video codec (faster)
            "-c:a", "aac",           # Convert audio to AAC
            "-shortest",             # Stop when shortest stream ends (video or audio)
            output_path
        ]
        
        logging.info(f"🎥 Creating video: looped_video.mp4 + script audio")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp audio
        if os.path.exists(tts_path):
            os.remove(tts_path)
        
        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logging.info(f"✅ Video created successfully: {output_path} ({file_size} bytes)")
            return True
        else:
            logging.error(f"❌ Video creation failed:")
            logging.error(f"Return code: {result.returncode}")
            logging.error(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Video creation error: {e}")
        return False

def translate_to_es_en_only(text):
    """Translate any input to Spanish and English ONLY - no Russian output"""
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{
                "role": "system", 
                "content": "You are a translator that ONLY outputs Spanish and English. No matter what language the input is (Russian, Chinese, etc.), translate it to BOTH Spanish and English. Format: 'Spanish: [translation]\nEnglish: [translation]'"
            }, {
                "role": "user", 
                "content": f"Translate this to Spanish and English only:\n\n{text}"
            }],
            "max_tokens": 500
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"Spanish: {text}\nEnglish: {text}"
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return f"Spanish: {text}\nEnglish: {text}"

def convert_audio_format_enhanced(input_path, output_path):
    """Enhanced audio conversion with better error handling"""
    try:
        # Enhanced ffmpeg command with better audio processing
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-acodec", "mp3",
            "-ar", "16000",  # 16kHz sample rate for Whisper
            "-ac", "1",      # Convert to mono
            "-ab", "128k",   # Set bitrate
            "-f", "mp3",     # Force MP3 format
            output_path
        ]
        
        logging.info(f"🔄 Running conversion command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logging.info(f"✅ Audio conversion successful: {output_path}")
            return True
        else:
            logging.error(f"❌ Audio conversion failed:")
            logging.error(f"Return code: {result.returncode}")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ Audio conversion timed out")
        return False
    except Exception as e:
        logging.error(f"❌ Audio conversion error: {e}")
        return False

def transcribe_audio_with_openai(audio_file_path):
    """Transcribe audio file using OpenAI Whisper with enhanced error handling"""
    try:
        # Verify file exists and has content
        if not os.path.exists(audio_file_path):
            logging.error(f"Audio file does not exist: {audio_file_path}")
            return None
            
        file_size = os.path.getsize(audio_file_path)
        if file_size < 100:
            logging.error(f"Audio file too small: {file_size} bytes")
            return None
            
        logging.info(f"📤 Sending {file_size} byte audio file to Whisper API...")
        
        # Enhanced transcription with better error handling
        with open(audio_file_path, 'rb') as f:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={
                    "model": "whisper-1",
                    "response_format": "json"
                },
                timeout=120  # Increased timeout for larger files
            )
        
        logging.info(f"📥 Whisper API response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            transcription = result.get("text", "").strip()
            
            if transcription:
                logging.info(f"🎤 Transcribed ({len(transcription)} chars): {transcription[:100]}...")
                return transcription
            else:
                logging.error("Empty transcription returned")
                return None
        else:
            logging.error(f"Whisper API error: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

def detect_user_country(message_text, user_id):
    """Detect which country the user is asking about or living in"""
    message_lower = message_text.lower()
    
    # Direct country mentions
    for country, context in COUNTRY_CONTEXTS.items():
        if country in message_lower:
            return country
        
        # Check for landmarks
        for landmark in context["landmarks"]:
            if landmark.lower() in message_lower:
                return country
        
        # Check for food mentions
        for food in context["food"]:
            if food.lower() in message_lower:
                return country
        
        # Check for currency mentions
        if context["currency"].lower() in message_lower:
            return country
    
    # City-based detection
    city_mappings = {
        "ciudad de méxico": "mexico", "cdmx": "mexico", "guadalajara": "mexico", "cancún": "mexico",
        "san josé": "costa_rica", "manuel antonio": "costa_rica", "tamarindo": "costa_rica",
        "bogotá": "colombia", "medellín": "colombia", "cartagena": "colombia", "cali": "colombia",
        "santiago": "chile", "valparaíso": "chile", "viña del mar": "chile",
        "buenos aires": "argentina", "córdoba": "argentina", "mendoza": "argentina",
        "san salvador": "el_salvador",
        "madrid": "spain", "barcelona": "spain", "valencia": "spain", "sevilla": "spain",
        "lima": "peru", "cusco": "peru", "arequipa": "peru"
    }
    
    for city, country in city_mappings.items():
        if city in message_lower:
            return country
    
    # Default to Panama if no specific country detected
    return "panama"

def process_message_with_claude_enhanced(user_id, message_text):
    """Enhanced processing with multi-country support - NO RUSSIAN IN OUTPUT"""
    try:
        # Detect the country context
        detected_country = detect_user_country(message_text, user_id)
        country_context = COUNTRY_CONTEXTS[detected_country]
        
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Build country-specific context
        context_info = f"""
COUNTRY CONTEXT: {detected_country.upper()}
Currency: {country_context['currency']}
Transport: {', '.join(country_context['transport'])}
Key Places: {', '.join(country_context['landmarks'])}
Local Food: {', '.join(country_context['food'])}
Common Greetings: {', '.join(country_context['greetings'])}
Cultural Tip: {country_context['cultural_tips']}
Climate: {country_context['climate']}
Expat Areas: {', '.join(country_context['expat_areas'])}
Language Note: {country_context['language_note']}
"""
        
        system_prompt = f"""You are Espaluz, a bilingual AI language tutor for expat families in Latin America and Spain.

{context_info}

CRITICAL: You MUST respond ONLY in Spanish and English. NEVER use Russian, Chinese, or any other language in your response.

Your answer must have TWO PARTS:

1️⃣ A COMPLETE, RICH, BILINGUAL response (Spanish and English ONLY):
   - Be warm and family-oriented for ALL family members
   - Address different ages (children, parents, grandparents)
   - Include specific cultural insights about {detected_country.upper()}
   - Use both Spanish and English naturally throughout
   - Be educational but conversational and emotionally supportive
   - Include practical tips relevant to {detected_country} specifically
   - Reference local customs, food, transport, and expat life
   - Make it interesting for ALL ages
   - Length: 700-1000 characters for rich, meaningful content
   - NO truncation - provide the FULL response

2️⃣ A SHORT video script (30-second spoken video) inside [VIDEO SCRIPT START] ... [VIDEO SCRIPT END]:
   - EXACTLY 2-3 short sentences that fit in 30 seconds
   - Use both Spanish and English
   - Warm, clear, and simple for spoken delivery
   - Include the KEY point about {detected_country}
   - Maximum 150 characters
   
Example structure:
Main response: Detailed, warm bilingual advice with cultural context for {detected_country}...
[VIDEO SCRIPT START]
¡Hola familia! En {detected_country}, [key tip]. Hello family! In {detected_country}, [key tip].
[VIDEO SCRIPT END]

Today is {datetime.now().strftime('%Y-%m-%d')}. Respond ONLY in Spanish and English."""

        messages = [{"role": "user", "content": message_text}]
        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": messages,
            "system": system_prompt,
            "max_tokens": 1500,  # Increased to prevent truncation
            "temperature": 0.7
        }
        
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        
        if r.status_code == 200:
            reply = r.json()["content"][0]["text"]
            full_reply = reply.strip()
            
            # Log the full reply to debug truncation
            logging.info(f"🤖 Full Claude reply length: {len(full_reply)} chars")
            logging.info(f"🤖 Full reply preview: {full_reply[:200]}...")
            
            # Extract video script
            short_reply = extract_video_script(full_reply)
            
            # Clean for speech
            full_reply_clean = clean_text_for_speech(full_reply)
            short_reply_clean = clean_text_for_speech(short_reply)
            
            logging.info(f"🌎 Generated response for {detected_country.upper()}")
            logging.info(f"📹 Video script: {short_reply_clean}")
            
            return {
                "full_reply": full_reply,
                "short_reply": short_reply,
                "full_reply_clean": full_reply_clean,
                "short_reply_clean": short_reply_clean,
                "detected_country": detected_country
            }
        else:
            logging.error(f"Claude API error: {r.status_code} - {r.text}")
            return fallback_response_enhanced(message_text, detected_country)
            
    except Exception as e:
        logging.error(f"Claude error: {e}")
        return fallback_response_enhanced(message_text, detected_country)

def fallback_response_enhanced(message_text, country="panama"):
    """Enhanced fallback with country context - NO RUSSIAN"""
    country_context = COUNTRY_CONTEXTS.get(country, COUNTRY_CONTEXTS["panama"])
    
    fallback_text = f"¡Hola familia! Te ayudo con español en {country.title()}. Hello family! I help you with Spanish in {country.title()}. {country_context['cultural_tips']} ¿En qué más puedo ayudarte? What else can I help you with?"
    
    video_script = f"¡Hola! Soy Espaluz en {country.title()}. Hello! I'm Espaluz in {country.title()}."
    
    return {
        "full_reply": fallback_text,
        "short_reply": video_script,
        "full_reply_clean": fallback_text,
        "short_reply_clean": video_script,
        "detected_country": country
    }

def process_whatsapp_voice_async_enhanced(user_id, media_url):
    """Enhanced voice processing with multi-country support - NO RUSSIAN OUTPUT"""
    def process():
        try:
            logging.info(f"🎤 Starting enhanced voice processing for {user_id}")
            
            # Send initial processing message
            try:
                send_whatsapp_message(user_id, "🎤 Procesando mensaje de voz... / Processing voice message...")
            except Exception as e:
                logging.error(f"Failed to send processing message: {e}")
            
            # Step 1: Download and convert audio
            logging.info(f"🔗 Getting authenticated media URL from: {media_url}")
            auth = HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            logging.info(f"📥 Downloading audio from authenticated URL...")
            download_response = requests.get(media_url, auth=auth, timeout=60)
            
            if download_response.status_code != 200:
                logging.error(f"Audio download failed: {download_response.status_code}")
                send_whatsapp_message(user_id, "❌ No pude descargar el audio. Could not download audio.")
                return
            
            logging.info(f"✅ Audio downloaded successfully, size: {len(download_response.content)} bytes")
            
            if len(download_response.content) < 100:
                logging.error(f"Downloaded content too small: {len(download_response.content)} bytes")
                send_whatsapp_message(user_id, "❌ Archivo de audio demasiado pequeño. Audio file too small.")
                return
            
            # Step 2: Save and convert audio
            timestamp = int(time.time())
            original_audio = f"/tmp/voice_original_{timestamp}.ogg"
            converted_audio = f"/tmp/voice_converted_{timestamp}.mp3"
            
            with open(original_audio, 'wb') as f:
                f.write(download_response.content)
            
            logging.info(f"💾 Audio saved to: {original_audio}")
            
            if not os.path.exists(original_audio) or os.path.getsize(original_audio) < 100:
                logging.error(f"Failed to save audio file properly")
                send_whatsapp_message(user_id, "❌ Error guardando archivo de audio. Error saving audio file.")
                return
            
            # Step 3: Convert audio format for Whisper
            logging.info("🔄 Converting audio format for transcription...")
            if not convert_audio_format_enhanced(original_audio, converted_audio):
                send_whatsapp_message(user_id, "❌ Error convirtiendo audio. Error converting audio.")
                return
            
            logging.info(f"✅ Audio converted to: {converted_audio}")
            
            # Step 4: Transcribe audio
            logging.info("🗣️ Transcribing with Whisper...")
            transcription = transcribe_audio_with_openai(converted_audio)
            
            if not transcription or len(transcription.strip()) == 0:
                logging.error("Transcription failed or empty")
                send_whatsapp_message(user_id, "❌ No pude transcribir el audio. Could not transcribe audio.")
                return
            
            logging.info(f"✅ Transcription successful: {transcription}")
            
            # Step 5: Send original transcription
            try:
                send_whatsapp_message(user_id, f"🗣️ Dijiste / You said:\n{transcription}")
            except Exception as e:
                logging.error(f"Failed to send transcription: {e}")
            
            # Step 6: Send translation (Spanish/English ONLY)
            try:
                translation = translate_to_es_en_only(transcription)
                send_whatsapp_message(user_id, f"📝 Traducción / Translation:\n{translation}")
            except Exception as e:
                logging.error(f"Failed to send translation: {e}")
            
            # Step 7: Get Claude's ENHANCED response with country detection
            result = process_message_with_claude_enhanced(user_id, transcription)
            detected_country = result.get('detected_country', 'panama')
            
            # Add country flag to response
            country_flags = {
                "panama": "🇵🇦", "mexico": "🇲🇽", "costa_rica": "🇨🇷", 
                "colombia": "🇨🇴", "chile": "🇨🇱", "argentina": "🇦🇷",
                "el_salvador": "🇸🇻", "spain": "🇪🇸", "peru": "🇵🇪"
            }
            flag = country_flags.get(detected_country, "🌎")
            
            # Step 8: Send FULL, country-specific text response
            try:
                # Split long messages to avoid truncation
                full_response = f"🤖 Espaluz {flag}:\n{result['full_reply']}"
                if len(full_response) > 1500:
                    # Split into chunks
                    chunks = [full_response[i:i+1400] for i in range(0, len(full_response), 1400)]
                    for i, chunk in enumerate(chunks, 1):
                        chunk_msg = f"📱 Parte {i}/{len(chunks)}:\n{chunk}" if len(chunks) > 1 else chunk
                        send_whatsapp_message(user_id, chunk_msg)
                        time.sleep(1)  # Brief pause between chunks
                else:
                    send_whatsapp_message(user_id, full_response)
                    
                logging.info(f"✅ Full response sent ({len(result['full_reply'])} chars)")
            except Exception as e:
                logging.error(f"Failed to send full response: {e}")
            
            # Step 9: Generate FULL voice message (country-aware) - with error handling
            try:
                audio_path = f"/tmp/reply_audio_{int(time.time())}.mp3"
                if generate_tts_audio(result["full_reply_clean"], audio_path):
                    if send_whatsapp_media(user_id, audio_path, "audio"):
                        logging.info(f"🎧 Country-specific audio sent for {detected_country}")
                    else:
                        logging.error(f"Failed to send audio media")
                else:
                    logging.error(f"Failed to generate TTS audio")
            except Exception as e:
                logging.error(f"Error with audio generation/sending: {e}")
            
            # Step 10: Generate SHORT video with country context - with error handling
            try:
                video_path = f"/tmp/espaluz_video_{int(time.time())}.mp4"
                if create_video_with_looped_base(result["short_reply_clean"], video_path):
                    if send_whatsapp_media(user_id, video_path, "video"):
                        logging.info(f"🎥 Country-specific video sent for {detected_country}")
                    else:
                        logging.error(f"Failed to send video media")
                else:
                    logging.error(f"Failed to create video")
            except Exception as e:
                logging.error(f"Error with video generation/sending: {e}")
            
            # Step 11: Clean up temp files
            for file_path in [original_audio, converted_audio]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logging.error(f"Failed to remove {file_path}: {e}")
                        
            logging.info("🧹 Voice processing cleanup completed")
                    
        except Exception as e:
            logging.exception(f"❌ Error processing voice message: {e}")
            try:
                send_whatsapp_message(user_id, "❌ Error procesando mensaje de voz. Error processing voice message.")
            except:
                logging.error("Failed to send error message")
    
    thread = threading.Thread(target=process, daemon=True)
    thread.start()
    logging.info(f"🚀 Enhanced voice processing thread started for {user_id}")

def process_whatsapp_message_async_enhanced(user_id, user_text):
    """Enhanced text processing with multi-country support - NO RUSSIAN OUTPUT"""
    def process():
        try:
            logging.info(f"📩 Processing enhanced message from {user_id}: {user_text}")
            
            # Step 1: Show received message
            try:
                send_whatsapp_message(user_id, f"📝 Recibí tu mensaje / I received your message:\n{user_text}")
            except Exception as e:
                logging.error(f"Failed to send received message: {e}")
            
            # Step 2: Send translation (Spanish/English ONLY)
            try:
                translation = translate_to_es_en_only(user_text)
                send_whatsapp_message(user_id, f"📝 Traducción / Translation:\n{translation}")
            except Exception as e:
                logging.error(f"Failed to send translation: {e}")
            
            # Step 3: Get Claude's ENHANCED response with country detection
            result = process_message_with_claude_enhanced(user_id, user_text)
            detected_country = result.get('detected_country', 'panama')
            
            # Add country flag to response
            country_flags = {
                "panama": "🇵🇦", "mexico": "🇲🇽", "costa_rica": "🇨🇷", 
                "colombia": "🇨🇴", "chile": "🇨🇱", "argentina": "🇦🇷",
                "el_salvador": "🇸🇻", "spain": "🇪🇸", "peru": "🇵🇪"
            }
            flag = country_flags.get(detected_country, "🌎")
            
            # Step 4: Send FULL, country-specific text response
            try:
                # Split long messages to avoid truncation
                full_response = f"🤖 Espaluz {flag}:\n{result['full_reply']}"
                if len(full_response) > 1500:
                    # Split into chunks
                    chunks = [full_response[i:i+1400] for i in range(0, len(full_response), 1400)]
                    for i, chunk in enumerate(chunks, 1):
                        chunk_msg = f"📱 Parte {i}/{len(chunks)}:\n{chunk}" if len(chunks) > 1 else chunk
                        send_whatsapp_message(user_id, chunk_msg)
                        time.sleep(1)  # Brief pause between chunks
                else:
                    send_whatsapp_message(user_id, full_response)
                    
                logging.info(f"✅ Full response sent ({len(result['full_reply'])} chars)")
            except Exception as e:
                logging.error(f"Failed to send full response: {e}")
            
            # Step 5: Generate FULL voice message (country-aware) - with error handling
            try:
                audio_path = f"/tmp/reply_audio_{int(time.time())}.mp3"
                if generate_tts_audio(result['full_reply_clean'], audio_path):
                    if send_whatsapp_media(user_id, audio_path, "audio"):
                        logging.info(f"🎧 Country-specific audio sent for {detected_country}")
                    else:
                        logging.error(f"Failed to send audio media")
                else:
                    logging.error(f"Failed to generate TTS audio")
            except Exception as e:
                logging.error(f"Error with audio generation/sending: {e}")
            
            # Step 6: Generate SHORT video with country context - with error handling
            try:
                video_path = f"/tmp/espaluz_video_{int(time.time())}.mp4"
                if create_video_with_looped_base(result['short_reply_clean'], video_path):
                    if send_whatsapp_media(user_id, video_path, "video"):
                        logging.info(f"🎥 Country-specific video sent for {detected_country}")
                    else:
                        logging.error(f"Failed to send video media")
                else:
                    logging.error(f"Failed to create video")
            except Exception as e:
                logging.error(f"Error with video generation/sending: {e}")
                
        except Exception as e:
            logging.error(f"Error in enhanced async processing: {e}")
            try:
                send_whatsapp_message(user_id, "Lo siento, hubo un error. Sorry, there was an error.")
            except:
                logging.error("Failed to send error message")
    
    thread = threading.Thread(target=process, daemon=True)
    thread.start()

def process_image_with_gpt4_vision(image_bytes):
    """Process image using GPT-4 Vision - Enhanced OCR - NO RUSSIAN OUTPUT"""
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
                    Then provide translations to ONLY Spanish and English with cultural context for expat families in Latin America.
                    NEVER output Russian or any other language - ONLY Spanish and English."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ALL text from this image. Preserve structure and formatting. Then translate to ONLY Spanish and English (no other languages) with cultural insights for expat families in Latin America."
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

def send_whatsapp_message(to, text):
    """Send text message via Twilio with enhanced error handling"""
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            'From': TWILIO_NUMBER,
            'To': f"whatsapp:{to}",
            'Body': text[:1500]  # Ensure we don't exceed WhatsApp limits
        }
        res = requests.post(url, data=data, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        logging.info(f"📤 Sent text to {to} status: {res.status_code}")
        
        if res.status_code == 201:
            return True
        elif res.status_code == 429:
            logging.error(f"Rate limit exceeded: {res.text}")
            return False
        else:
            logging.error(f"Text message error: {res.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error sending text message: {e}")
        return False

def send_whatsapp_media(to, file_path, media_type="audio"):
    """Send media file via hosted URL with enhanced error handling"""
    try:
        # Verify file exists and has content
        if not os.path.exists(file_path):
            logging.error(f"Media file does not exist: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        if file_size < 100:
            logging.error(f"Media file too small: {file_size} bytes")
            return False
            
        logging.info(f"📤 Preparing to send {media_type} file ({file_size} bytes)")
        
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
        
        if res.status_code == 201:
            logging.info(f"✅ {media_type.title()} sent successfully")
            success = True
        elif res.status_code == 429:
            logging.error(f"❌ Rate limit exceeded for {media_type}: {res.text}")
            success = False
        else:
            logging.error(f"❌ {media_type.title()} send failed: {res.text}")
            success = False
        
        # Clean up original file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return success
            
    except Exception as e:
        logging.error(f"Error sending {media_type}: {e}")
        return False

def process_whatsapp_image_async(user_id, media_url):
    """Process image message asynchronously - NO RUSSIAN OUTPUT"""
    def process():
        try:
            # Download image with authentication
            auth = HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            response = requests.get(media_url, auth=auth)
            
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
def handle_message_enhanced():
    """Enhanced webhook handler with better error handling"""
    try:
        user_id = request.form.get("From", "").replace("whatsapp:", "")
        user_text = request.form.get("Body", "")
        media_count = int(request.form.get("NumMedia", 0))

        if not user_id:
            return "ok", 200

        logging.info(f"🌎 Enhanced message from {user_id}: media_count={media_count}, text='{user_text[:50]}'")

        # Handle media messages
        if media_count > 0:
            media_url = request.form.get("MediaUrl0", "")
            media_type = request.form.get("MediaContentType0", "")
            
            logging.info(f"📎 Media received - Type: {media_type}, URL: {media_url}")
            
            if 'image' in media_type:
                process_whatsapp_image_async(user_id, media_url)
            elif 'audio' in media_type or 'ogg' in media_type:
                logging.info(f"🎤 Enhanced voice message detected from {user_id}")
                process_whatsapp_voice_async_enhanced(user_id, media_url)
            else:
                send_whatsapp_message(user_id, f"📄 Recibí tu archivo ({media_type}). I received your file.")
            
            return "ok", 200

        # Handle text messages with enhanced processing
        if not user_text:
            return "ok", 200

        process_whatsapp_message_async_enhanced(user_id, user_text)
        return "ok", 200

    except Exception as e:
        logging.exception("❌ Error in enhanced webhook")
        return "ok", 500

@app.route("/", methods=["GET"])
def health_enhanced():
    return jsonify({
        "status": "running",
        "bot": "Espaluz WhatsApp - Fixed Multi-Country Edition",
        "version": "v5.1-fixed-no-russian",
        "supported_countries": list(COUNTRY_CONTEXTS.keys()),
        "features": ["text", "voice", "video", "image_processing", "voice_transcription", "claude_ai", "multi_country_support"],
        "fixes": {
            "no_russian_output": "All responses strictly in Spanish and English only",
            "full_responses": "Increased max_tokens to prevent truncation",
            "chunked_messages": "Long messages split into chunks",
            "error_handling": "Enhanced error handling for media sending",
            "rate_limiting": "Better handling of Twilio rate limits"
        },
        "country_detection": "automatic based on landmarks, food, cities, and cultural references",
        "message_sequence": {
            "voice": ["processing", "original_transcription", "translation_es_en_only", "full_country_specific_response", "audio", "video"],
            "text": ["received_confirmation", "translation_es_en_only", "full_country_specific_response", "audio", "video"],
            "image": ["processing", "text_extraction", "translation_es_en_only", "cultural_context"]
        },
        "countries": {
            "panama": "🇵🇦 USD, Metro, Casco Viejo",
            "mexico": "🇲🇽 Peso, CDMX Metro, Zócalo",
            "costa_rica": "🇨🇷 Colón, Pura Vida, Arenal",
            "colombia": "🇨🇴 Peso, TransMilenio, Cartagena",
            "chile": "🇨🇱 Peso, Metro, Valparaíso",
            "argentina": "🇦🇷 Peso, Subte, Buenos Aires",
            "el_salvador": "🇸🇻 USD, Pupusas, El Tunco",
            "spain": "🇪🇸 Euro, AVE, Sagrada Familia",
            "peru": "🇵🇪 Sol, Machu Picchu, Ceviche"
        },
        "endpoints": {
            "webhook": "/webhook",
            "media": "/media/<filename>",
            "health": "/"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🚀 Starting FIXED Espaluz WhatsApp Bot - Multi-Country Edition on port {port}")
    logging.info("✅ Key Fixes Applied:")
    logging.info("   🚫 NO Russian text in responses - Spanish/English only")
    logging.info("   📝 Full responses - increased max_tokens to prevent truncation")
    logging.info("   📱 Message chunking for long responses")
    logging.info("   🛡️  Enhanced error handling for media sending")
    logging.info("   ⏱️  Better rate limit handling")
    logging.info("")
    logging.info("✅ Features enabled:")
    logging.info("   📝 Text messages (Spanish, English only)")
    logging.info("   🎤 Voice message transcription and processing")
    logging.info("   📷 Image text recognition and translation")
    logging.info("   🤖 Claude AI responses with country detection")
    logging.info("   🎥 Video generation with looped_video.mp4")
    logging.info("   🎧 Audio responses")
    logging.info("   🌍 Multi-language translation (to Spanish/English only)")
    logging.info("   🌎 Multi-country support for Latin America & Spain")
    app.run(host="0.0.0.0", port=port)