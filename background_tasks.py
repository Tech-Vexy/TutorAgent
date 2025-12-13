import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, Callable, Coroutine

logger = logging.getLogger("BackgroundTaskManager")

class BackgroundTaskManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackgroundTaskManager, cls).__new__(cls)
            cls._instance._tasks = {} # type: Dict[str, Dict[str, Any]]
        return cls._instance

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task."""
        return self._tasks.get(task_id, {"status": "not_found", "result": None})

    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        return self._tasks

    def submit_task(self, coro_func: Callable[..., Coroutine], *args, **kwargs) -> str:
        """
        Submit a coroutine function to be run in the background.
        Returns a task_id immediately.
        """
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "type": coro_func.__name__
        }
        
        # Create the asyncio task
        asyncio.create_task(self._run_task(task_id, coro_func, *args, **kwargs))
        
        return task_id

    async def _run_task(self, task_id: str, coro_func: Callable, *args, **kwargs):
        try:
            logger.info(f"Starting background task {task_id}")
            result = await coro_func(*args, **kwargs)
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["result"] = result
            logger.info(f"Background task {task_id} completed")
        except Exception as e:
            logger.error(f"Background task {task_id} failed: {e}")
            self._tasks[task_id]["status"] = "failed"
            self._tasks[task_id]["error"] = str(e)

# Global instance
background_manager = BackgroundTaskManager()
