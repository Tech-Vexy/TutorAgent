import os
import logging

logger = logging.getLogger(__name__)

def get_groq_drive_tool(auth_token: str = None):
    """
    Returns the Groq Google Drive MCP Connector definition.
    Reference: https://console.groq.com/docs/tool-use/remote-mcp/connectors#google-drive-example
    """
    token = auth_token or os.getenv("GOOGLE_DRIVE_OAUTH_TOKEN")
    
    if not token:
        logger.warning("GOOGLE_DRIVE_OAUTH_TOKEN not set. Google Drive MCP connector will be disabled.")
        return None
        
    return {
        "type": "mcp",
        "server_label": "Google Drive",
        "connector_id": "connector_googledrive",
        "authorization": token,
        "require_approval": "never" # or "always"
    }

def get_groq_gmail_tool(auth_token: str = None):
    """
    Returns the Groq Gmail MCP Connector definition.
    """
    token = auth_token or os.getenv("GMAIL_OAUTH_TOKEN")
    
    if not token:
        return None
        
    return {
        "type": "mcp",
        "server_label": "Gmail",
        "connector_id": "connector_gmail",
        "authorization": token,
        "require_approval": "never"
    }
