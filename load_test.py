import asyncio
import websockets
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("LoadTest")

URI = "ws://localhost:8080/ws/chat"
NUM_CLIENTS = 20
DURATION = 10

async def simulate_client(client_id):
    try:
        async with websockets.connect(URI) as websocket:
            logger.info(f"Client {client_id} connected")
            
            # Send a simple message
            message = {
                "type": "user_message",
                "content": "Hello, this is a load test.",
                "user_id": f"user_{client_id}",
                "thread_id": f"thread_{client_id}",
                "model": "llama-3.1-70b-versatile" # Use a fast model if possible, or mock
            }
            await websocket.send(json.dumps(message))
            
            # Wait for response or timeout
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(response)
                    if data.get("type") == "end_turn":
                        logger.info(f"Client {client_id} finished turn")
                        break
            except asyncio.TimeoutError:
                logger.error(f"Client {client_id} timed out")
                
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")

async def main():
    logger.info(f"Starting load test with {NUM_CLIENTS} clients...")
    start_time = time.time()
    
    tasks = [simulate_client(i) for i in range(NUM_CLIENTS)]
    await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    logger.info(f"Load test completed in {duration:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
