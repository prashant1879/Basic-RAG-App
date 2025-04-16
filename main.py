import asyncio
from logger import logger
from dotenv import load_dotenv
from api_tools import get_vector_store, invoke_model, init_chatbot
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_chatbot()
    await get_vector_store()
    yield

# Initialize the FastAPI app
app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    phone_number: str = None
    question: str = None
    
@app.post("/chat/start")
async def start_chat(request: ChatRequest):
    phone_number = request.phone_number
    content = request.question
    try:
        message = await invoke_model(content, phone_number)        
        return {"phone_number": phone_number, "message":message}
    except Exception as e:
        logger.info(e)
        logger.info("Exception at /chat/start api {e}")

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=5000)
    # how to run
    # python main.py