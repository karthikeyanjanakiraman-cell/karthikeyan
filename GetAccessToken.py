import os
import json
import base64
import smtplib
import ssl
from email.message import EmailMessage

from fyers_apiv3 import fyersModel


def decode_jwt_and_extract_auth_code(jwt_token):
    """
    Decode JWT token to verify structure and extract claims
    """
    try:
        parts = jwt_token.split(".")
        print(f"\n📊 JWT Structure:")
        print(f"  Number of parts: {len(parts)}")

        if len(parts) != 3:
            print(f"  ❌ Invalid JWT format. Expected 3 parts, got {len(parts)}")
            return None

        print(f"  ✅ Valid JWT structure (header.payload.signature)")

        payload = parts[1]
        print(f"  Payload length: {len(payload)} characters")

        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
            print(f"  Added {padding} padding characters")

        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)

        print(f"\n📋 Decoded JWT Payload:")
        print(json.dumps(data, indent=2))

        if data.get("sub") == "auth_code":
            print(f"\n✅ CONFIRMED: JWT is an authorization token")
            print(f"  'sub' claim = 'auth_code' ✓")
            app_id = data.get("app_id", "Unknown")
            print(f"  app_id in JWT: {app_id}")
            return data
        else:
            print(f"\n❓ 'sub' claim is not 'auth_code'")
            print(f"  sub value: {data.get('sub')}")
            return data

    except Exception as e:
        print(f"❌ Error decoding JWT: {e}")
        import traceback

        traceback.print_exc()
        return None


def send_access_token_email(access_token: str):
    """
    Send the access token to the configured recipient using SMTP.
    All values are expected to come from environment variables.
    """
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if not (recipient_email and sender_email and sender_password):
        print("\n⚠️ Email not sent: missing RECIPIENT_EMAIL / SENDER_EMAIL / SENDER_PASSWORD env vars.")
        return

    try:
        subject = "Fyers Access Token"
        body = f"Your latest Fyers access token:\n\n{access_token}"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("\n✅ Access token emailed successfully.")
    except Exception as e:
        print(f"\n❌ Failed to send email: {e}")


def get_access_token():
    """
    Main function to get access token from Fyers API
    All credentials come from environment variables.
    """

    # ---------- Load credentials from env ----------
    client_id = os.environ["CLIENT_ID"]
    secret_key = os.environ["SECRET_KEY"]
    redirect_uri = os.environ["REDIRECT_URI"]
    auth_code_input = os.environ["AUTH_CODE"]

    print("\n" + "=" * 70)
    print("DEBUGGING INFO")
    print("=" * 70)
    print(f"Client ID: {client_id}")
    print(f"Redirect URI: {redirect_uri}")
    print(f"Auth Code Input Length: {len(auth_code_input)} characters")

    # ---------- JWT / auth code handling ----------
    has_eyj = auth_code_input.startswith("eyJ")
    has_dots = auth_code_input.count(".") >= 2
    is_long = len(auth_code_input) > 100

    print(f"\n🔍 Auth Code Analysis:")
    print(f"  Starts with 'eyJ': {has_eyj}")
    print(f"  Contains dots (JWT separator): {has_dots}")
    print(f"  Length > 100 chars: {is_long}")
    print(f"  First 60 chars: {auth_code_input[:60]}...")

    is_jwt = has_eyj and has_dots and is_long

    if is_jwt:
        print(f"\n⚠️ DETECTED: Input is a JWT token")
        print(f"✅ The JWT itself IS the auth code!")
        print(f"  Decoding to verify structure...")

        jwt_data = decode_jwt_and_extract_auth_code(auth_code_input)

        if jwt_data and jwt_data.get("sub") == "auth_code":
            auth_code = auth_code_input
            print(f"\n✅ JWT verified as valid auth code")
            print(f"  Using full JWT as auth code")
            print(f"  Length: {len(auth_code)} characters")

            jwt_app_id = jwt_data.get("app_id", "")
            config_app_id = client_id.split("-")[0] if "-" in client_id else client_id

            print(f"\n🔐 Credential Validation:")
            print(f"  app_id in JWT: {jwt_app_id}")
            print(f"  app_id from CLIENT_ID: {config_app_id}")

            if jwt_app_id == config_app_id:
                print(f"  ✅ MATCH! Credentials are correct")
            else:
                print(f"  ⚠️ MISMATCH! Check if you're using the right credentials")
                print(f"  JWT was generated for app_id: {jwt_app_id}")
                print(f"  But CLIENT_ID is: {client_id}")
        else:
            print(f"\n❌ JWT does not appear to be a valid auth code")
            print(f"  'sub' claim should be 'auth_code'")
            return False
    else:
        auth_code = auth_code_input
        print(f"\n✅ Using auth code (short code)")
        print(f"  Length: {len(auth_code)} characters")

    print("=" * 70)

    # ---------- Fyers Session and token ----------
    try:
        print(f"\n📡 Initializing Fyers SessionModel...")
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        print(f"✅ SessionModel created successfully")

        print(f"\n🔐 Setting authorization code...")
        session.set_token(auth_code)
        print(f"✅ Auth code set (length: {len(auth_code)})")

        print(f"\n⏳ Requesting access token from Fyers API...")
        print(f"  This may take a few seconds...")
        response = session.generate_token()

        print(f"\n📡 Full Response:")
        print(json.dumps(response, indent=2))

        if isinstance(response, dict) and "access_token" in response:
            access_token = response["access_token"]

            print(f"\n{'=' * 70}")
            print(f"✅ SUCCESS! Access Token received")
            print(f"{'=' * 70}")
            print(f"Access Token (first 50 chars): {access_token[:50]}...")

            # Email token
            send_access_token_email(access_token)

            return True

        elif isinstance(response, dict) and "message" in response:
            error_code = response.get("code", "Unknown")
            error_msg = response.get("message", "Unknown error")

            print(f"\n❌ ERROR {error_code}: {error_msg}")

            if error_code == -437:
                print(f"\n💡 Troubleshooting 'invalid auth code' error:")
                print(f"\n  🔍 Check these:")
                print(f"  1. JWT 'app_id' should match your Client ID (before -100)")
                print(f"  2. The JWT itself IS the auth code (not a field within it)")
                print(f"  3. JWT 'sub' claim should be 'auth_code'")
                print(f"  4. JWT might have expired (exp claim shows expiration time)")
                print(f"  5. Make sure you're using the correct Client ID and Secret")
                print(f"\n  ✅ To generate fresh auth code:")
                print(f"  1. Visit: https://api-hcrealtime.fyers.in/api/v3/generate-authcode")
                print(f"  2. Click 'Get AuthCode' button")
                print(f"  3. You'll be redirected - capture the full page content")
                print(f"  4. The JWT in the 'authorization code' field is what you need")
                print(f"  5. Paste entire JWT into AUTH_CODE env var")
                print(f"  6. Run this script immediately (JWTs expire!)")

            return False

        else:
            print(f"\n❌ Unexpected response format")
            print(f"Response: {response}")
            return False

    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Exception Type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = get_access_token()
    exit(0 if success else 1)
