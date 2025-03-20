from fastapi import FastAPI, Request
from openai import OpenAI, AsyncOpenAI
from fastapi.responses import StreamingResponse
import uvicorn
import json

app = FastAPI()

internal_client = AsyncOpenAI(
    api_key="dummy-key",
    base_url="http://your_ip/v1"
)

async def stream_generator(response):
    try:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.choices[0].delta.content}}]})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.json()
    
    try:
        if body.get("stream", False):
            response = await internal_client.chat.completions.create(
                model=body.get("model", "olympiad"),
                messages=body.get("messages", []),
                temperature=body.get("temperature", 1.0),
                max_tokens=body.get("max_tokens", 512),
                stream=True,
                presence_penalty=body.get("presence_penalty", 0),
                frequency_penalty=body.get("frequency_penalty", 0),
                top_p=body.get("top_p", 1),
                stop=body.get("stop", None),
                n=body.get("n", 1),
                user=body.get("user", None)
            )
            
            return StreamingResponse(
                stream_generator(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            response = await internal_client.chat.completions.create(
                model=body.get("model", "olympiad"),
                messages=body.get("messages", []),
                temperature=body.get("temperature", 1.0),
                max_tokens=body.get("max_tokens", 512),
                stream=False,
                presence_penalty=body.get("presence_penalty", 0),
                frequency_penalty=body.get("frequency_penalty", 0),
                top_p=body.get("top_p", 1),
                stop=body.get("stop", None),
                n=body.get("n", 1),
                user=body.get("user", None)
            )
            
            return response
            
    except Exception as e:
        return {"error": str(e), "details": str(type(e))}


if __name__ == "__main__":
    uvicorn.run(app,
        host="0.0.0.0",
        port=8001,
        limit_concurrency=100
    )

