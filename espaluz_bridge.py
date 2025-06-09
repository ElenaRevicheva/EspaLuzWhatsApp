from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
from requests.auth import HTTPBasicAuth

from core_logic import process_message_internal  # Claude logic

# App setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Meta webhook verification (GET) - Works for both Meta and Twilio
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "espaluz123":
        logging.info("✅ Webhook verified.")
        return challenge, 200
    else:
        logging.warning("❌ Webhook verification failed.")
        return "Verification failed", 403

# Handle incoming WhatsApp messages (POST) - FIXED FOR TWILIO
@app.route("/webhook", methods=["POST"])
def handle_message():
    try:
        # Check if it's Twilio (form data) or Meta (JSON)
        content_type = request.headers.get('Content-Type', '')
        logging.info(f"🌐 Incoming webhook, Content-Type: {content_type}")
        
        if 'application/x-www-form-urlencoded' in content_type:
            # Twilio format (form data)
            logging.info("📱 Processing Twilio webhook")
            user_id = request.form.get('From', '').replace('whatsapp:', '')
            user_text = request.form.get('Body', '')
            
            if not user_id or not user_text:
                logging.info("📭 No valid Twilio message data.")
                return "ok", 200
                
            logging.info(f"📩 Twilio message from {user_id}: {user_text}")
            
            # Process message
            reply = process_message_internal(user_id, user_text)
            reply_text = reply.get("response", "Lo siento, no entendí eso.")
            
            # Send via Twilio
            send_whatsapp_message_twilio(user_id, reply_text)
            return "ok", 200
            
        else:
            # Meta format (JSON) - Original code
            logging.info("📱 Processing Meta webhook")
            data = request.get_json()
            logging.info(f"🌐 Meta webhook: {data}")

            entry = data.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])

            if not messages:
                logging.info("📭 No Meta message found.")
                return "ok", 200

            message = messages[0]
            user_id = message.get("from")
            user_text = message.get("text", {}).get("body", "")

            logging.info(f"📩 Meta message from {user_id}: {user_text}")

            reply = process_message_internal(user_id, user_text)
            reply_text = reply.get("response", "Lo siento, no entendí eso.")

            send_whatsapp_message_meta(user_id, reply_text)
            return "ok", 200

    except Exception as e:
        logging.exception("❌ Error handling message")
        return jsonify({"error": str(e)}), 500

# Send WhatsApp reply via Twilio
def send_whatsapp_message_twilio(to_number, message_text):
    """Send WhatsApp message using Twilio API"""
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        
        data = {
            'From': 'whatsapp:+14155238886',
            'To': f'whatsapp:{to_number}',
            'Body': message_text[:1600]  # Twilio limit
        }
        
        response = requests.post(
            url, 
            data=data, 
            auth=HTTPBasicAuth(account_sid, auth_token)
        )
        
        logging.info(f"📤 Twilio sent to {to_number}, response: {response.status_code}")
        if response.status_code != 201:
            logging.error(f"Twilio error: {response.text}")
            
    except Exception as e:
        logging.error(f"❌ Error sending Twilio message: {e}")

# Send WhatsApp reply via Meta (original function)
def send_whatsapp_message_meta(to_number, message_text):
    """Send WhatsApp message using Meta API"""
    try:
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
        logging.info(f"📤 Meta sent to {to_number}, response: {response.status_code} - {response.text}")
        
    except Exception as e:
        logging.error(f"❌ Error sending Meta message: {e}")

# Backward compatibility
def send_whatsapp_message(to_number, message_text):
    """Backward compatibility - defaults to Twilio for now"""
    send_whatsapp_message_twilio(to_number, message_text)

# Health check endpoint
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Espaluz WhatsApp Bridge",
        "version": "2.0-twilio-fixed"
    })

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"🚀 Starting Espaluz WhatsApp Bridge on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)