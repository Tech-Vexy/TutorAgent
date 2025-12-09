
import os
import io
import pickle
import tempfile
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveConnector:
    def __init__(self, credentials_path="drive_credentials.json", service_account_path="service_account.json", token_path="token.pickle"):
        self.credentials_path = credentials_path
        self.service_account_path = service_account_path
        self.token_path = token_path
        self.service = None
        
    def authenticate(self):
        """Authenticates using Service Account (priority) or User OAuth."""
        creds = None
        
        # 1. Try Service Account (Preferred for App Backend)
        if os.path.exists(self.service_account_path):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    self.service_account_path, scopes=SCOPES)
                print("✅ Authenticated with Service Account")
            except Exception as e:
                print(f"⚠️ Service Account failed: {e}")

        # 2. Try User OAuth (Fallback)
        if not creds:
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_path):
                         # If neither exists, we can't connect
                         if not os.path.exists(self.service_account_path):
                            raise FileNotFoundError("No credentials found. Please provide 'service_account.json' (App) or 'drive_credentials.json' (User).")
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)
        return True

    def list_files(self, page_size=20, query=None):
        """Lists files from Google Drive."""
        if not self.service:
            self.authenticate()
            
        q = "trashed = false"
        if query:
            q += f" and name contains '{query}'"
            
        # Add support for Shared Drives if using Service Account
        results = self.service.files().list(
            pageSize=page_size, 
            fields="nextPageToken, files(id, name, mimeType, size)",
            q=q,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        return results.get('files', [])

    def download_and_ingest_streaming(self, file_id, file_name, knowledge_base):
        """
        Stream-downloads a file to a temp location and processes it chunk-by-chunk.
        Optimized for large files (Books, Past Papers).
        """
        if not self.service:
            self.authenticate()

        # Get Metadata
        meta = self.service.files().get(fileId=file_id).execute()
        mime_type = meta.get('mimeType')
        print(f"📥 Starting stream download for: {file_name} ({mime_type})")

        # Create Temp File
        suffix = ".pdf" if "pdf" in mime_type else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            request = self.service.files().get_media(fileId=file_id)
            
            # Streaming Download (Chunked)
            downloader = MediaIoBaseDownload(tmp_file, request, chunksize=1024*1024) # 1MB chunks
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                # We could log progress here: status.progress()
            
            tmp_path = tmp_file.name

        chunks_added = 0
        try:
            # Process based on type
            if "pdf" in mime_type or file_name.lower().endswith(".pdf"):
                chunks_added = self._process_pdf_streaming(tmp_path, file_name, knowledge_base)
            else:
                chunks_added = self._process_text_streaming(tmp_path, file_name, knowledge_base)
                
            return chunks_added, f"Successfully ingested {file_name} in chunks."
            
        except ImportError:
            return 0, "pypdf not installed."
        except Exception as e:
            return 0, f"Error processing file: {e}"
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _process_pdf_streaming(self, file_path, source_name, kb):
        import pypdf
        reader = pypdf.PdfReader(file_path)
        total_pages = len(reader.pages)
        
        buffer_text = ""
        count = 0
        
        # Process page by page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                buffer_text += text + "\n"
            
            # Ingest every 5 pages or 10k characters (Stream Ingestion)
            if len(buffer_text) > 10000 or (i == total_pages - 1 and len(buffer_text) > 500):
                kb.add_document(buffer_text, metadata={"source": source_name, "page_range": f"{i-5}-{i}"})
                buffer_text = ""
                count += 1
                
        return count

    def _process_text_streaming(self, file_path, source_name, kb):
        count = 0
        buffer_text = ""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                buffer_text += line
                if len(buffer_text) > 5000:
                    kb.add_document(buffer_text, metadata={"source": source_name})
                    buffer_text = ""
                    count += 1
            if buffer_text:
                kb.add_document(buffer_text, metadata={"source": source_name})
                count += 1
        return count

# Singleton instance
drive_connector = GoogleDriveConnector()
