class RegenerateRequest(BaseModel):
    model_preference: str = Field("fast", description="Model to use: 'fast' or 'smart'")
    user_id: str = Field("default_user", description="Unique identifier for the user")

@app.post("/threads/{thread_id}/regenerate", tags=["Chat"], summary="Regenerate the last AI response")
async def regenerate_endpoint(thread_id: str, request: Optional[RegenerateRequest] = None):
    """
    Regenerates the response for the last user message in the thread.
    """
    try:
        pool = app.state.pool
        last_user_msg = None
        user_id = "default_user"
        
        # 1. Fetch last user message
        async with pool.connection() as conn:
            cursor = await conn.execute("""
                SELECT content, user_id, created_at
                FROM thread_messages_v2
                WHERE thread_id = %s AND message_type = 'user'
                ORDER BY created_at DESC
                LIMIT 1
            """, (thread_id,))
            row = await cursor.fetchone()
            
            if not row:
                return {"error": "No user message found to regenerate from."}
            
            last_user_msg = row[0]
            user_id = row[1]
            msg_timestamp = row[2]
            
            # 2. Delete subsequent AI messages (cleanup bad responses)
            del_result = await conn.execute("""
                DELETE FROM thread_messages_v2
                WHERE thread_id = %s AND created_at > %s
            """, (thread_id, msg_timestamp))
            
            await conn.commit()
            logger.info(f"♻️ Regenerated cleanup: Deleted {del_result.rowcount} stale messages after {msg_timestamp}.")

    except Exception as e:
        logger.error(f"Error in regenerate setup: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")


    logger.info(f"♻️ Regenerating response for thread {thread_id}, user {user_id}. Query: {last_user_msg[:30]}...")

    # 3. Stream new response with alternative explanation
    async def event_generator():
        # Modify the user's message to request a different explanation approach
        regeneration_instruction = (
            f"{last_user_msg}\\n\\n"
            f"[System Note: Please provide an alternative explanation using a different teaching method or perspective. "
            f"Try a different approach than your previous response - use different examples, analogies, or teaching styles.]"
        )
        
        # Profile
        user_profile = await get_or_create_profile(user_id)
        
        # Toggle model: if last was fast, use smart; if last was smart, use fast
        default_model = "smart" if request and request.model_preference == "fast" else "fast"
        model_pref = request.model_preference if request else default_model
        
        # Message with regeneration context
        human_message = HumanMessage(content=regeneration_instruction)
        
        # Config
        run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}
        
        # Run Graph with GLOBAL workflow
        checkpointer = AsyncPostgresSaver(pool)
        app_graph = workflow.compile(checkpointer=checkpointer)
        
        logger.info(f"⚡ Starting regeneration with alternative teaching approach... Model: {model_pref}")
        
        logger.info("⚡ Entering astream_events loop...")
        step_count = 0
        async for event in app_graph.astream_events(
            {
                "messages": [human_message],
                "model_preference": model_pref,
                "user_profile": user_profile, 
                "tool_invocations": 0
            },
            config=run_config,
            version="v1"
        ):
            step_count += 1
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name in ["planner", "reflection"]:
                    continue
                
                content = event["data"]["chunk"].content
                if content:
                    chunk_json = {'type': 'message', 'content': content}
                    yield f"data: {json.dumps(chunk_json)}\\n\\n"
                    # Broadcast to WS
                    await manager.broadcast_to_thread(thread_id, chunk_json)
            
            elif kind == "on_tool_end":
                tool_output = event["data"].get("output")
                if tool_output and isinstance(tool_output, str) and "[IMAGE_GENERATED_BASE64_DATA:" in tool_output:
                     chunk_json = {'type': 'token', 'content': tool_output}
                     yield f"data: {json.dumps(chunk_json)}\\n\\n"
                     await manager.broadcast_to_thread(thread_id, chunk_json)
                     
        logger.info("⚡ Regeneration completed successfully.")
        yield "data: [DONE]\\n\\n"
        await manager.broadcast_to_thread(thread_id, {"type": "end_turn"})
    
    # Wrap to save
    async def saving_generator():
        logger.info("⚡ Starting streaming response generator...")
        full_response = ""
        try:
            async for chunk in event_generator():
                yield chunk
                if "content" in chunk:
                    try:
                        parts = chunk.split("data: ", 1)
                        if len(parts) > 1:
                            data_str = parts[1].strip()
                            if data_str != "[DONE]":
                                 json_data = json.loads(data_str)
                                 if json_data.get("type") == "message":
                                     full_response += json_data.get("content", "")
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error in streaming generator: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"
        
        # Save final AI response
        if full_response:
             try:
                await _save_message_db(thread_id, user_id, full_response, "ai")
             except Exception as e:
                logger.error(f"Failed to save regenerated message: {e}")

    return StreamingResponse(saving_generator(), media_type="text/event-stream")

async def _save_message_db(thread_id: str, user_id: str, content: str, msg_type: str):
    """Helper to save messages to the SQL storage (V2 with UUIDs)."""
    import uuid
    try:
        pool = app.state.pool
        async with pool.connection() as conn:
            msg_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO thread_messages_v2 (id, thread_id, user_id, message_type, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (msg_id, thread_id, user_id, msg_type, content))
    except Exception as e:
        logger.error(f"Error internal save_message: {e}")

