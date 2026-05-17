from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram

from api.deps import init_services
from api.routes import ner, translate, stream

REQUEST_COUNT = Counter("turing_tag_requests_total", "Total requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("turing_tag_request_seconds", "Request latency", ["endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_services()
    yield


app = FastAPI(title="turing_tag", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(ner.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(stream.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    from api.deps import _ner_service
    return {
        "model_type": _ner_service.model_type if _ner_service else None,
        "translation_backend": "google",
    }
