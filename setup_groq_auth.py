import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriveAuth")

# Scopes required by Groq Drive Connector
# Reference: https://console.groq.com/docs/tool-use/remote-mcp/connectors#google-drive-example
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

CREDENTIALS_FILE = "path_to_your_credentials.json" # Users often download as 'credentials.json' or 'client_secret_....json'

def find_credentials():
    """Look for common credential file names."""
    files = [f for f in os.listdir(".") if f.endswith(".json")]
    for f in files:
        if "secret" in f.lower() or "credential" in f.lower():
            return f
    return None

def main():
    print("🚀 Google Drive Connector Setup for Groq")
    print("---------------------------------------")
    
    creds_file = find_credentials()
    
    if not creds_file:
        print("❌ No 'credentials.json' or client secret file found in this directory.")
        print("\nTo fix this:")
        print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
        print("2. Create a project and enable 'Google Drive API'.")
        print("3. Go to 'Credentials' > 'Create Credentials' > 'OAuth client ID' (Desktop App).")
        print("4. Download the JSON file, save it here as 'credentials.json', and run this script again.")
        return

    print(f"✅ Found credentials file: {creds_file}")
    
    creds = None
    # Token file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("🔐 Launching browser for authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Now we have the credentials, specificially the ACCESS TOKEN
    # for the Groq Connector.
    access_token = creds.token
    print("\n🎉 Authentication Successful!")
    print(f"🔑 Access Token: {access_token}")
    
    # Optional: Update .env automatically
    update = input("\nDo you want to add this to your .env file? (y/n): ")
    if update.lower() == 'y':
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        
        # Remove existing key if present
        lines = [l for l in lines if not l.startswith("GOOGLE_DRIVE_OAUTH_TOKEN=")]
        
        # Add new key
        lines.append(f"GOOGLE_DRIVE_OAUTH_TOKEN={access_token}\n")
        
        with open(env_path, 'w') as f:
            f.writelines(lines)
        print("✅ .env file updated. Restart the agent to use Google Drive!")
    else:
        print("\nMake sure to set GOOGLE_DRIVE_OAUTH_TOKEN in your environment variables.")

if __name__ == "__main__":
    main()
