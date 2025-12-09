import sys
import asyncio
import os
import uvicorn

def main():
    # Force SelectorEventLoop on Windows for psycopg compatibility
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            print("Set WindowsSelectorEventLoopPolicy.")
        except Exception as e:
            print(f"Failed to set event loop policy: {e}")

    # Use monkeypatching to prevent uvicorn from overriding the policy via 'auto' loop setup
    # Uvicorn 0.30+ might behave differently, but patching Config.setup_event_loop is generally robust.
    
    # We must import the app *after* setting the policy if possible, 
    # but more importantly, we control the server execution.
    
    # Define a custom run function that keeps our loop
    # actually, if we use uvicorn.run(..., loop="asyncio") it might use the global policy?
    # No, uvicorn's "asyncio" loop setup calls asyncio.new_event_loop().
    
    import server  # Verify import works and applies any top-level side effects
    import socket

    def find_free_port(start_port=8080, max_port=8100):
        for port in range(start_port, max_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        return 0 # Let OS choose

    port = find_free_port()
    if port == 0:
        # If range exhausted, let uvicorn choose random (port=0) but we want to know it
        # Actually uvicorn doesn't natively return the port easily if we pass 0 in Config without running
        # But for now let's just use what we found or 0
        pass

    print(f"Starting server on port {port}")

    # We will run uvicorn programmatically
    config = uvicorn.Config("server:app", host="0.0.0.0", port=port, reload=True)
    
    # Patch the setup_event_loop method on the instance or class
    # We'll patch the config instance method
    original_setup = config.setup_event_loop
    
    def setup_event_loop_override():
        if sys.platform == "win32":
            loop = asyncio.get_event_loop_policy().get_event_loop()
            # If current loop is Proactor, we have a problem. 
            # But get_event_loop() might verify policy.
            pass # Trust the global policy we set earlier
        else:
            original_setup()

    config.setup_event_loop = setup_event_loop_override
    
    server_instance = uvicorn.Server(config)
    server_instance.run()

if __name__ == "__main__":
    main()
