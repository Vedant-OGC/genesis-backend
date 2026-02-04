from fastapi.responses import StreamingResponse
import time

from api import GenesisRequest, app
from app import run_genesis_stream

@app.post("/genesis/stream")
def genesis_stream(req: GenesisRequest):

    def event_generator():
        for chunk in run_genesis_stream(req.idea):
            yield f"data: {chunk}\n\n"
            time.sleep(0.02)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
