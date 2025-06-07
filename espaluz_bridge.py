from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests

from core_logic import process_message_internal  # Claude logic

# App setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Meta webhook verification (GET)
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "espaluz123":
        logging.info("✅ Webhook verified by Meta.")
        return challenge, 200
    else:
        logging.warning("❌ Webhook verification failed.")
        return "Verification failed", 403

# Handle incoming WhatsApp messages (POST)
@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        data = request.get_json()
        logging.info(f"🌐 Incoming webhook: {data}")

        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            logging.info("📭 No user message found.")
            return "ok", 200

        message = messages[0]
        user_id = message.get("from")
        user_text = message.get("text", {}).get("body", "")

        logging.info(f"📩 Message from {user_id}: {user_text}")

        reply = process_message_internal(user_id, user_text)
        reply_text = reply.get("response", "Lo siento, no entendí eso.")

        send_whatsapp_message(user_id, reply_text)
        return "ok", 200

    except Exception as e:
        logging.exception("❌ Error handling message")
        return jsonify({"error": str(e)}), 500

# Send WhatsApp reply
def send_whatsapp_message(to_number, message_text):
    phone_number_id = os.getenv("PHONE_NUMBER_ID")
    access_token = os.getenv("WHATSAPP_API_TOKEN")

    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": { "body": message_text }
    }

    response = requests.post(url, headers=headers, json=payload)
    logging.info(f"📤 Sent to {to_number}, response: {response.status_code} - {response.text}")

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
