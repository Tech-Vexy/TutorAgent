import requests
import sys

def verify_cors():
    url = "http://localhost:8080/health"
    origin = "http://localhost:33689" # Example ephemeral port
    
    print(f"Testing CORS for {url} with Origin: {origin}")
    
    try:
        response = requests.get(url, headers={"Origin": origin})
        
        print(f"Status Code: {response.status_code}")
        print("Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
            
        allow_origin = response.headers.get("access-control-allow-origin")
        allow_credentials = response.headers.get("access-control-allow-credentials")
        
        if allow_origin == origin and allow_credentials == "true":
            print("\nSUCCESS: CORS headers are correct.")
            return True
        else:
            print(f"\nFAILURE: Incorrect CORS headers.")
            print(f"Expected Access-Control-Allow-Origin: {origin}")
            print(f"Got: {allow_origin}")
            print(f"Expected Access-Control-Allow-Credentials: true")
            print(f"Got: {allow_credentials}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to server at http://localhost:8080. Is it running?")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

if __name__ == "__main__":
    if verify_cors():
        sys.exit(0)
    else:
        sys.exit(1)
