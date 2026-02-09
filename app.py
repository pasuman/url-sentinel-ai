"""
FastAPI service for phishing URL detection.

Exposes the multi-view LightGBM ensemble via REST endpoints.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from inference import classify_url, load_ensemble


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ensemble()
    yield


app = FastAPI(
    title="url-police-ai",
    description="Phishing URL detection using multi-view LightGBM ensemble",
    version="1.0.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    url: str
    network_features: dict[str, float] | None = None  # Optional: 14 network features


class ClassifyResponse(BaseModel):
    url: str
    phishing_probability: float
    is_phishing: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_single(req: ClassifyRequest):
    """
    Classify a URL for phishing.

    Network features are optional. If not provided, they default to -1 (missing).
    Network features: time_response, domain_spf, asn_ip, time_domain_activation,
    time_domain_expiration, qty_ip_resolved, qty_nameservers, qty_mx_servers,
    ttl_hostname, tls_ssl_certificate, qty_redirects, url_google_index,
    domain_google_index, url_shortened.
    """
    prob = classify_url(req.url, network_features=req.network_features)
    return ClassifyResponse(
        url=req.url,
        phishing_probability=round(prob, 4),
        is_phishing=prob >= 0.5,
    )
