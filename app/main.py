import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router

app = FastAPI(
    title="Lyrics Synchronization Service",
    description="API for synchronizing lyrics with media files using WhisperX.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":  
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000)     