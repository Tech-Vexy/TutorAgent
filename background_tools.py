import asyncio
import time
from langchain_core.tools import tool
from background_tasks import background_manager

# --- Simulation of a heavy function ---
async def _heavy_study_guide_generator(topic: str, pages: int = 5):
    """(Simulated) Generates a heavy PDF study guide."""
    # Simulate work
    total_time = 10  # seconds
    steps = 5
    for i in range(steps):
        await asyncio.sleep(total_time / steps)
        # We could update progress here if the manager supported it
    
    return f"Study Guide for '{topic}' ({pages} pages) generated successfully at /tmp/guide_{int(time.time())}.pdf"

async def _heavy_web_scrape(url: str):
    """(Simulated) Scrapes a large documentation site."""
    await asyncio.sleep(5)
    return f"Scraping of {url} completed. 150 pages indexed."

# --- Tools exposed to the Agent ---

@tool
async def generate_study_guide_tool(topic: str):
    """
    Starts a background job to generate a comprehensive study guide PDF.
    Use this for large request (e.g. '50 page guide').
    Returns a Task ID immediately.
    """
    task_id = background_manager.submit_task(_heavy_study_guide_generator, topic, pages=50)
    return f"Started generating study guide for '{topic}'. Task ID: {task_id}. You can check status with 'check_task_status'."

@tool
async def scrape_docs_tool(url: str):
    """
    Starts a background job to scrape a website.
    Returns a Task ID immediately.
    """
    task_id = background_manager.submit_task(_heavy_web_scrape, url)
    return f"Started scraping {url}. Task ID: {task_id}. Use 'check_task_status' to monitor."

@tool
async def check_task_status(task_id: str):
    """
    Checks the status of a background task given its ID.
    Returns the status (running/completed) and result if done.
    """
    status = background_manager.get_task_status(task_id)
    if status["status"] == "not_found":
        return "Task not found. Please check the ID."
    elif status["status"] == "completed":
        return f"Task Completed! Result: {status['result']}"
    elif status["status"] == "failed":
        return f"Task Failed. Error: {status['error']}"
    else:
        return f"Task is currently {status['status']}..."
