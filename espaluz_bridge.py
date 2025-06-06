from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

from core_logic import process_message_internal  # Claude brain or shared logic

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enable basic logging
logging.basicConfig(level=logging.INFO)

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

# Launch server on assigned Railway port or default to 5000
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
