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
        "to": user_id,  # This must be a full WhatsApp number like "50764597519"
        "type": "template",
        "template": {
            "name": "espaluz_reply",
            "language": { "code": "es_MX" },  # Use your approved locale, or "es_PA" if supported
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

    response = requests.post(
        f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
        headers=headers,
        json=payload
    )

    return {
        "user_id": user_id,
        "status_code": response.status_code,
        "result": response.json() if response.content else "No response"
    }
