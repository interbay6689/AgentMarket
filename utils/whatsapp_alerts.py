# utils/whatsapp_alerts.py
from twilio.rest import Client

account_sid = 'AC09c9469594d0ea559a2be4e1cd6bff44'
auth_token = '54cb39a27bc771883ff91b3a69366662'
client = Client(account_sid, auth_token)

def send_whatsapp_message(body, to_whatsapp_number):
    from_whatsapp_number = 'whatsapp:+14155238886'  # מספר Twilio Sandbox

    message = client.messages.create(
        from_=from_whatsapp_number,
        body=body,
        to=f'whatsapp:{to_whatsapp_number}'
    )
    return message.sid
