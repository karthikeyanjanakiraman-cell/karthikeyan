import configparser
import json
import base64
from fyers_apiv3 import fyersModel

def decode_jwt_and_extract_auth_code(jwt_token):
    """
    Decode JWT token to verify structure and extract claims
    """
    try:
        parts = jwt_token.split('.')
        
        print(f"\n📊 JWT Structure:")
        print(f"   Number of parts: {len(parts)}")
        
        if len(parts) != 3:
            print(f"   ❌ Invalid JWT format. Expected 3 parts, got {len(parts)}")
            return None
        
        print(f"   ✅ Valid JWT structure (header.payload.signature)")
        
        # Decode the payload (add padding if needed)
        payload = parts[1]
        print(f"   Payload length: {len(payload)} characters")
        
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
            print(f"   Added {padding} padding characters")
        
        try:
            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            
            print(f"\n📋 Decoded JWT Payload:")
            print(json.dumps(data, indent=2))
            
            # Check if this is an auth code JWT (sub claim = "auth_code")
            if data.get('sub') == 'auth_code':
                print(f"\n✅ CONFIRMED: JWT is an authorization token")
                print(f"   'sub' claim = 'auth_code' ✓")
                app_id = data.get('app_id', 'Unknown')
                print(f"   app_id in JWT: {app_id}")
                return data
            else:
                print(f"\n❓ 'sub' claim is not 'auth_code'")
                print(f"   sub value: {data.get('sub')}")
                return data
            
        except json.JSONDecodeError as je:
            print(f"   ❌ JSON decode error: {je}")
            return None
            
    except Exception as e:
        print(f"❌ Error decoding JWT: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_access_token():
    """
    Main function to get access token from Fyers API
    """
    # Load config file
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    
    client_id = config['fyers_credentials']['client_id'].strip()
    secret_key = config['fyers_credentials']['secret_key'].strip()
    redirect_uri = config['fyers_credentials']['redirect_uri'].strip()
    auth_code_input = config['fyers_credentials']['auth_code'].strip()
    
    # Debug: Print credentials (censored)
    print("\n" + "="*70)
    print("DEBUGGING INFO")
    print("="*70)
    print(f"Client ID: {client_id}")
    print(f"Redirect URI: {redirect_uri}")
    print(f"Auth Code Input Length: {len(auth_code_input)} characters")
    
    # Better JWT detection
    has_eyj = auth_code_input.startswith('eyJ')
    has_dots = auth_code_input.count('.') >= 2
    is_long = len(auth_code_input) > 100
    
    print(f"\n🔍 Auth Code Analysis:")
    print(f"   Starts with 'eyJ': {has_eyj}")
    print(f"   Contains dots (JWT separator): {has_dots}")
    print(f"   Length > 100 chars: {is_long}")
    print(f"   First 60 chars: {auth_code_input[:60]}...")
    
    # Detect JWT
    is_jwt = has_eyj and has_dots and is_long
    
    if is_jwt:
        print(f"\n⚠️  DETECTED: Input is a JWT token (589 chars)")
        print(f"✅ The JWT itself IS the auth code!")
        print(f"   Decoding to verify structure...")
        
        jwt_data = decode_jwt_and_extract_auth_code(auth_code_input)
        
        if jwt_data and jwt_data.get('sub') == 'auth_code':
            auth_code = auth_code_input
            print(f"\n✅ JWT verified as valid auth code")
            print(f"   Using full JWT as auth code")
            print(f"   Length: {len(auth_code)} characters")
            
            # Extract app_id from JWT for validation
            jwt_app_id = jwt_data.get('app_id', '')
            config_app_id = client_id.split('-')[0] if '-' in client_id else client_id
            
            print(f"\n🔐 Credential Validation:")
            print(f"   app_id in JWT: {jwt_app_id}")
            print(f"   app_id in config (base): {config_app_id}")
            
            if jwt_app_id == config_app_id:
                print(f"   ✅ MATCH! Credentials are correct")
            else:
                print(f"   ⚠️  MISMATCH! Check if you're using the right credentials")
                print(f"      JWT was generated for app_id: {jwt_app_id}")
                print(f"      But config has: {client_id}")
        else:
            print(f"\n❌ JWT does not appear to be a valid auth code")
            print(f"   'sub' claim should be 'auth_code'")
            return False
    else:
        auth_code = auth_code_input
        print(f"\n✅ Using auth code from config (short code)")
        print(f"   Length: {len(auth_code)} characters")
    
    print("="*70)
    
    # Create session model
    try:
        print(f"\n📡 Initializing Fyers SessionModel...")
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        print(f"✅ SessionModel created successfully")
        
        # Set the auth code
        print(f"\n🔐 Setting authorization code...")
        session.set_token(auth_code)
        print(f"✅ Auth code set (length: {len(auth_code)})")
        
        # Generate access token
        print(f"\n⏳ Requesting access token from Fyers API...")
        print(f"   This may take a few seconds...")
        response = session.generate_token()
        
        # Print the full response
        print(f"\n📡 Full Response:")
        print(json.dumps(response, indent=2))
        
        # Handle success
        if isinstance(response, dict) and 'access_token' in response:
            access_token = response['access_token']
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS! Access Token received")
            print(f"{'='*70}")
            print(f"Access Token: {access_token[:50]}...")
            
            # Save to config
            config['fyers_credentials']['access_token'] = access_token
            with open('config.ini', 'w') as f:
                config.write(f)
            print(f"\n✅ Access token saved to config.ini")
            print(f"{'='*70}")
            return True
        
        # Handle errors
        elif isinstance(response, dict) and 'message' in response:
            error_code = response.get('code', 'Unknown')
            error_msg = response.get('message', 'Unknown error')
            print(f"\n❌ ERROR {error_code}: {error_msg}")
            
            if error_code == -437:
                print(f"\n💡 Troubleshooting 'invalid auth code' error:")
                print(f"\n  🔍 Check these:")
                print(f"     1. JWT 'app_id' should match your Client ID (before -100)")
                print(f"     2. The JWT itself IS the auth code (not a field within it)")
                print(f"     3. JWT 'sub' claim should be 'auth_code'")
                print(f"     4. JWT might have expired (exp claim shows expiration time)")
                print(f"     5. Make sure you're using the correct Client ID and Secret")
                print(f"\n  ✅ To generate fresh auth code:")
                print(f"     1. Visit: https://api-hcrealtime.fyers.in/api/v3/generate-authcode")
                print(f"     2. Click 'Get AuthCode' button")
                print(f"     3. You'll be redirected - capture the full page content")
                print(f"     4. The JWT in the 'authorization code' field is what you need")
                print(f"     5. Paste entire JWT into config.ini auth_code field")
                print(f"     6. Run this script immediately (JWTs expire!)")
            
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
