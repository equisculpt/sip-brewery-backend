import logging
import requests
import json
# from websocket import create_connection  # Uncomment if using websockets

logger = logging.getLogger("notification_service")

# --- WhatsApp Notification (Twilio API example) ---
def send_whatsapp(user_number, message):
    # Replace with your Twilio credentials and endpoint
    TWILIO_SID = "YOUR_TWILIO_SID"
    TWILIO_TOKEN = "YOUR_TWILIO_TOKEN"
    FROM_NUMBER = "whatsapp:+YOUR_TWILIO_NUMBER"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
    data = {
        "From": FROM_NUMBER,
        "To": f"whatsapp:+{user_number}",
        "Body": message
    }
    try:
        resp = requests.post(url, data=data, auth=(TWILIO_SID, TWILIO_TOKEN))
        logger.info(f"WhatsApp notification sent to {user_number}: {resp.status_code}")
    except Exception as e:
        logger.error(f"WhatsApp notification failed: {e}")

# --- Web/App Notification (WebSocket example) ---
def send_websocket(user_id, message):
    # Example: send to a running websocket server
    # ws = create_connection("ws://localhost:8765")
    # ws.send(json.dumps({"user_id": user_id, "message": message}))
    # ws.close()
    logger.info(f"Websocket notification to {user_id}: {message}")

# --- Email Notification (SMTP example) ---
def send_email(email_address, subject, message):
    import smtplib
    from email.mime.text import MIMEText
    SMTP_SERVER = "smtp.example.com"
    SMTP_PORT = 587
    SMTP_USER = "YOUR_EMAIL"
    SMTP_PASS = "YOUR_PASS"
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = email_address
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [email_address], msg.as_string())
        logger.info(f"Email sent to {email_address}")
    except Exception as e:
        logger.error(f"Email notification failed: {e}")

# --- Unified Notification Interface ---
def notify_user(user_id, channel, message, meta=None):
    if channel == "whatsapp":
        send_whatsapp(user_id, message)
    elif channel == "web":
        send_websocket(user_id, message)
    elif channel == "email":
        send_email(user_id, meta.get("email", ""), message)
    else:
        logger.warning(f"Unknown channel: {channel}")
