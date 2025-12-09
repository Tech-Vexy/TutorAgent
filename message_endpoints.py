"""
Add endpoint to save individual messages to thread.
This ensures messages persist even if WebSocket disconnects.
"""
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Literal

class SaveMessageRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str
    message_type: Literal["user", "ai"]

@app.post("/threads/{thread_id}/save_message")
async def save_message_to_thread(thread_id: str, request: SaveMessageRequest):
    """
    Save a single message to a thread.
    This allows incremental saving during streaming.
    """
    try:
        pool = app.state.pool
        
        # Save to user_threads metadata
        await upsert_thread_metadata(
            thread_id=request.thread_id,
            user_id=request.user_id,
            title=request.message[:50] if request.message_type == "user" else None
        )
        
        # Also save to a messages table for quick retrieval
        async with pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_messages (
                    id SERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_messages_thread_id 
                ON thread_messages(thread_id, created_at)
            """)
            
            await conn.execute("""
                INSERT INTO thread_messages (thread_id, user_id, message_type, content)
                VALUES (%s, %s, %s, %s)
            """, (request.thread_id, request.user_id, request.message_type, request.message))
            
        return {"status": "saved"}
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/messages_direct")
async def get_thread_messages_direct(thread_id: str):
    """
    Get messages from thread_messages table (faster, more reliable).
    Falls back to checkpoint if no messages found.
    """
    try:
        pool = app.state.pool
        
        async with pool.connection() as conn:
            # Check if table exists
            cursor = await conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'thread_messages'
                )
            """)
            table_exists = (await cursor.fetchone())[0]
            
            if table_exists:
                cursor = await conn.execute("""
                    SELECT message_type, content, created_at
                    FROM thread_messages
                    WHERE thread_id = %s
                    ORDER BY created_at ASC
                """, (thread_id,))
                
                rows = await cursor.fetchall()
                
                if rows:
                    return [{
                        "type": row[0],
                        "content": row[1],
                        "timestamp": row[2].isoformat() if row[2] else None
                    } for row in rows]
        
        # Fallback to checkpoint method
        return await get_thread_messages(thread_id)
        
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return []
