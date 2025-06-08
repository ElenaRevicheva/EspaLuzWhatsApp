import requests
import os

def process_message_internal(user_id, user_message):
    """Process WhatsApp messages using the Telegram bot's logic"""
    
    # Import the main bot's internal processing function
    from main import process_message_internal as telegram_processor
    
    # Create mock user/chat info for WhatsApp users
    user_info = {
        "id": user_id,
        "first_name": f"WhatsApp_{user_id[-4:]}",  # Use last 4 digits as name
        "username": None,
        "language_code": "es"  # Default to Spanish
    }
    
    chat_info = {
        "id": user_id,
        "type": "private"
    }
    
    # Process through the same logic as Telegram
    result = telegram_processor(user_id, user_message, user_info, chat_info)
    
    # Now send the response back via WhatsApp API
    phone_number_id = os.getenv("PHONE_NUMBER_ID")
    access_token = os.getenv("WHATSAPP_API_TOKEN")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Send text message
    text_payload = {
        "messaging_product": "whatsapp",
        "to": user_id,
        "type": "text",
        "text": {
            "body": result["full_reply"][:4096]  # WhatsApp has a 4096 char limit
        }
    }
    
    try:
        # Send main text response
        response = requests.post(
            f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
            headers=headers,
            json=text_payload
        )
        
        # Generate and send voice message
        if result.get("full_reply_clean"):
            from main import generate_tts_audio
            voice_path = f"/tmp/voice_{user_id}.mp3"
            
            if generate_tts_audio(result["full_reply_clean"], voice_path):
                # Upload audio to WhatsApp
                with open(voice_path, "rb") as f:
                    files = {"file": (f"audio.mp3", f, "audio/mpeg")}
                    upload_response = requests.post(
                        f"https://graph.facebook.com/v18.0/{phone_number_id}/media",
                        headers={"Authorization": f"Bearer {access_token}"},
                        files=files
                    )
                    
                    if upload_response.status_code == 200:
                        media_id = upload_response.json()["id"]
                        
                        # Send audio message
                        audio_payload = {
                            "messaging_product": "whatsapp",
                            "to": user_id,
                            "type": "audio",
                            "audio": {"id": media_id}
                        }
                        requests.post(
                            f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
                            headers=headers,
                            json=audio_payload
                        )
                
                os.remove(voice_path)
        
        # Generate and send video message
        if result.get("short_reply_clean"):
            from main import create_video_with_audio
            video_path = f"/tmp/video_{user_id}.mp4"
            
            if create_video_with_audio(result["short_reply_clean"], video_path):
                # Upload video to WhatsApp
                with open(video_path, "rb") as f:
                    files = {"file": (f"video.mp4", f, "video/mp4")}
                    upload_response = requests.post(
                        f"https://graph.facebook.com/v18.0/{phone_number_id}/media",
                        headers={"Authorization": f"Bearer {access_token}"},
                        files=files
                    )
                    
                    if upload_response.status_code == 200:
                        media_id = upload_response.json()["id"]
                        
                        # Send video message
                        video_payload = {
                            "messaging_product": "whatsapp",
                            "to": user_id,
                            "type": "video",
                            "video": {"id": media_id}
                        }
                        requests.post(
                            f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
                            headers=headers,
                            json=video_payload
                        )
                
                os.remove(video_path)
        
        return {
            "response": result["full_reply"],
            "status": "success"
        }
        
    except Exception as e:
        return {
            "response": "❌ Error sending WhatsApp message",
            "error": str(e)
        }
