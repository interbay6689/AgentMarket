import os
from twilio.rest import Client

account_sid = os.environ.get('TWILIO_ACCOUNT_SID', '')
auth_token  = os.environ.get('TWILIO_AUTH_TOKEN', '')

_client = None


def _get_client():
    global _client
    if _client is None:
        if not account_sid or not auth_token:
            raise RuntimeError(
                "Twilio credentials not set. "
                "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables."
            )
        _client = Client(account_sid, auth_token)
    return _client


def send_whatsapp_message(body: str, to_whatsapp_number: str) -> str:
    from_whatsapp_number = 'whatsapp:+14155238886'  # Twilio Sandbox number
    message = _get_client().messages.create(
        from_=from_whatsapp_number,
        body=body,
        to=f'whatsapp:{to_whatsapp_number}'
    )
    return message.sid
