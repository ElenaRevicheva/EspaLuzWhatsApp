from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

from core_logic import process_message_internal  # Claude logic here

# App setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Webhook verification (GET)
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

# Message handler (POST)
@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        data = request.get_json()

        user_id = data.get("user_id", "unknown")
        user_message = data.get("message", "")

        logging.info(f"📩 Incoming from {user_id}: {user_message}")

        response = process_message_internal(user_id, user_message)

        return jsonify(response), 200
    except Exception as e:
        logging.exception("❌ Error processing request")
        return jsonify({"error": str(e)}), 500

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
