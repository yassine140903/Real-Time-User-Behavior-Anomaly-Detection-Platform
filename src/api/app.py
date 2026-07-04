# src/api/app.py

import redis
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.config import REDIS_HOST, REDIS_PORT
from src.enrichment.core import EnrichmentCore
from src.scoring.core import ScoringCore
from src.scoring.fusion import ScoreFusion
from src.decision.core import DecisionCore
from src.decision.decision import DecisionService
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: wire dependencies ──────────────────────
    models_dir = Path("models")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    fusion = ScoreFusion(str(models_dir))
    engine = DecisionService()

    app.state.redis = r
    app.state.enrichment_core = EnrichmentCore(redis_client=r, models_dir=str(models_dir))
    app.state.scoring_core = ScoringCore(redis_client=r, fusion=fusion, models_dir=str(models_dir))
    app.state.decision_core = DecisionCore(redis_client=r, engine=engine)

    yield

    # ── Shutdown: clean up ──────────────────────────────
    r.close()


app = FastAPI(
    title="Amen Bank Anomaly Detection — Simulation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")