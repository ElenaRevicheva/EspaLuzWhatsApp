from flask import Flask, request, jsonify
from datetime import datetime
import base64
import uuid
import os

# Import your main logic
import main
from main import (
    process_message_internal,
    process_voice_internal,
    process_image_internal,
    get_user_progress_internal,
    generate_tts_audio,
    create_video_with_audio
)

app = Flask(__name__)

# === WhatsApp Unified Webhook ===
@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    data = request.json
    print(f"📥 Incoming WhatsApp message: {data}")

    user_id = data.get("user_id", str(uuid.uuid4()))
    user_info = data.get("user_info", {
        "id": user_id,
        "first_name": "User",
        "username": "whatsapp_user"  # ✅ This avoids AttributeError
    })
    chat_info = data.get("chat_info", {"id": user_id, "type": "private"})

    try:
        if "text" in data:
            message_text = data["text"]
            result = process_message_internal(user_id, message_text, user_info, chat_info)

        elif "audio_path" in data:
            audio_file_path = data["audio_path"]
            result = process_voice_internal(user_id, audio_file_path, user_info, chat_info)

        elif "image_bytes" in data:
            image_bytes = bytes.fromhex(data["image_bytes"])
            result = process_image_internal(user_id, image_bytes, user_info, chat_info)

        else:
            return jsonify({"error": "Unsupported message type"}), 400

        # Save audio
        audio_path = f"/tmp/{user_id}_voice.mp3"
        generate_tts_audio(result["full_reply_clean"], audio_path)

        # Create video
        video_path = f"/tmp/{user_id}_video.mp4"
        create_video_with_audio(result["short_reply_clean"], video_path)

        return jsonify({
            "translation": result["translation"],
            "full_reply": result["full_reply"],
            "short_reply": result["short_reply"],
            "thinking": result["thinking_process"],
            "audio_path": audio_path,
            "video_path": video_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Telegram-style fallback routes ===
@app.route("/process", methods=["POST"])
def process_text():
    data = request.get_json()
    user_id = data.get("user_id")
    message_text = data.get("text")
    user_info = data.get("user_info", {})
    chat_info = data.get("chat_info", {})

    if not user_id or not message_text:
        return jsonify({"error": "Missing user_id or text"}), 400

    try:
        result = process_message_internal(user_id, message_text, user_info, chat_info)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/voice", methods=["POST"])
def process_voice():
    user_id = request.form.get("user_id")
    user_info = request.form.get("user_info", {})
    chat_info = request.form.get("chat_info", {})

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file uploaded"}), 400

    temp_path = f"/tmp/{datetime.now().timestamp()}.mp3"
    audio_file.save(temp_path)

    try:
        result = process_voice_internal(user_id, temp_path, user_info, chat_info)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/image", methods=["POST"])
def process_image():
    data = request.get_json()
    user_id = data.get("user_id")
    image_base64 = data.get("image_base64")
    user_info = data.get("user_info", {})
    chat_info = data.get("chat_info", {})

    if not image_base64:
        return jsonify({"error": "Missing image_base64"}), 400

    image_bytes = base64.b64decode(image_base64)

    try:
        result = process_image_internal(user_id, image_bytes, user_info, chat_info)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/progress", methods=["GET"])
def get_progress():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        result = get_user_progress_internal(user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Launch Flask ===
if __name__ == "__main__":
    app.run(port=5000)
