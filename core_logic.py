import requests
import os

def process_message_internal(user_id, user_message):
    phone_number_id = os.getenv("PHONE_NUMBER_ID")
    access_token = os.getenv("WHATSAPP_API_TOKEN")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": user_id,  # Must be full WhatsApp number, e.g., "50764597519"
        "type": "template",
        "template": {
            "name": "espaluz_reply",
            "language": { "code": "es_MX" },  # Or "es_PA" if your template uses that
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        { "type": "text", "text": "¡Hola Elena! Hoy aprenderemos algo nuevo. 📘" }
                    ]
                }
            ]
        }
    }

    try:
        response = requests.post(
            f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
            headers=headers,
            json=payload
        )
        response_data = response.json() if response.content else {}

        return {
            "response": f"✅ WhatsApp message sent to {user_id}",
            "status_code": response.status_code,
            "result": response_data
        }

    except Exception as e:
        return {
            "response": "❌ Error sending WhatsApp message",
            "error": str(e)
        }
