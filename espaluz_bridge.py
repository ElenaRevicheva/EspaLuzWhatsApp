from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from core_logic import process_message_internal  # <-- this is your Claude brain

app = Flask(__name__)
CORS(app)

@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        data = request.get_json()
        user_id = data.get("user_id", "unknown")
        user_message = data.get("message", "")

        result = process_message_internal(user_id, user_message)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
