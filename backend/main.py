"""
FastAPI Main Application
Initializes the app and includes the API routers.
"""

from fastapi import FastAPI
from backend.api.v1 import routers as v1_routers
import os
import redis
from rq import Queue

# --- RQ Setup ---
# Use the environment variable set in docker-compose.yaml
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Redis connection and RQ Queue
try:
    redis_conn = redis.from_url(REDIS_URL)
    # The queue name must match the worker's queue name (alloskin-jobs)
    task_queue = Queue('alloskin-jobs', connection=redis_conn)
except Exception as e:
    # If Redis connection fails, we can't enqueue jobs, which will be caught in the health check
    print(f"FATAL: Could not connect to Redis: {e}")
    redis_conn = None
    task_queue = None
# --- End RQ Setup ---


app = FastAPI(
    title="AllosKin API",
    description="API for the AllosKin causal analysis pipeline.",
    version="0.1.0",
)

# Pass the queue object to the router so it can enqueue jobs
app.state.task_queue = task_queue
app.state.redis_conn = redis_conn 

# Include the v1 API routes
app.include_router(v1_routers.api_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint for health check.
    """
    return {"message": "Welcome to the AllosKin API. Go to /docs for details."}