"""
API Routers for V1
Defines endpoints for job submission, status polling, and result retrieval.
"""

import shutil
import json
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from typing import Dict, Any, List
import uuid
from pathlib import Path
from rq.job import Job
from redis import RedisError

# Import the master task function
from backend.tasks import run_analysis_job

api_router = APIRouter()

# --- Directory Definitions ---
UPLOAD_DIR = Path("/app/data/uploads")
RESULTS_DIR = Path("/app/data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Dependencies ---
def get_queue(request: Request):
    """Provides the RQ queue object."""
    if request.app.state.task_queue is None:
        raise HTTPException(status_code=503, detail="Worker queue not initialized. Check Redis connection.")
    return request.app.state.task_queue

# --- REFACTORED File Saving Helper ---
async def save_uploaded_files(
    files_dict: Dict[str, UploadFile], 
    job_folder: Path
) -> Dict[str, str]:
    """
    Saves uploaded files from a dictionary to a job-specific folder
    and returns a path dict for the worker.
    """
    saved_paths = {}
    
    # Map frontend form keys to worker path keys
    key_to_path_key = {
        "active_traj": "active_traj_path",
        "active_topo": "active_topo_path",
        "inactive_traj": "inactive_traj_path",
        "inactive_topo": "inactive_topo_path",
        "config": "config_path",
    }

    for key, file_obj in files_dict.items():
        if key not in key_to_path_key:
            print(f"Warning: Unknown file key '{key}' skipped.")
            continue 
        
        # Use the original filename for saving
        file_path = job_folder / file_obj.filename
        try:
            with open(file_path, "wb") as buffer:
                # Read in 1MB chunks
                while content := await file_obj.read(1024 * 1024): 
                    buffer.write(content)
            
            # Use the dict 'key' to build the path map for the worker
            path_key = key_to_path_key[key]
            saved_paths[path_key] = str(file_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file {file_obj.filename}: {e}")

    if len(saved_paths) != 5:
        # This check is still valid and important
        raise HTTPException(status_code=400, detail=f"File mapping failed. Mapped {len(saved_paths)}/5 required files. Check form field names.")
    
    return saved_paths

# --- Health Check Endpoint ---

@api_router.get("/health/check", summary="End-to-end system health check")
async def health_check(request: Request):
    # ... (existing health_check code remains unchanged)
    report = {
        "api_status": "ok",
        "redis_status": {"status": "unknown"},
        "worker_status": {"status": "unknown"},
    }
    
    # 1. Check Redis connection
    redis_conn = request.app.state.redis_conn
    if redis_conn:
        try:
            redis_conn.ping()
            report["redis_status"] = {"status": "ok", "info": "Connected and ping successful."}
        except RedisError as e:
            report["redis_status"] = {"status": "error", "error": f"Redis connection failed: {str(e)}"}
    else:
         report["redis_status"] = {"status": "error", "error": "Redis client failed to initialize."}
        
    # 2. Check Worker Queue status
    if report["redis_status"]["status"] == "ok":
        task_queue = request.app.state.task_queue
        try:
            report["worker_status"] = {"status": "ok", "queue_length": task_queue.count}
        except Exception as e:
            report["worker_status"] = {"status": "error", "error": f"Error interacting with RQ queue: {str(e)}"}

    if report["redis_status"]["status"] != "ok" or report["worker_status"]["status"] != "ok":
        raise HTTPException(status_code=503, detail=report)
    return report

# --- Job Status Endpoint ---

@api_router.get("/job/status/{job_id}", summary="Get the live status of a running job")
async def get_job_status(job_id: str, request: Request):
    """
    Polls RQ for the live status of an enqueued job.
    The job_id here is the RQ job ID, not our UUID.
    """
    task_queue = get_queue(request)
    try:
        job = Job.fetch(job_id, connection=task_queue.connection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found in RQ.")
        
    status = job.get_status()
    response = {
        "job_id": job_id,
        "status": status,
        "meta": job.meta,
    }
    
    if status == 'finished' or status == 'failed':
        # The return value of the task (`result_payload`)
        response["result"] = job.result
    
    return response

# --- REFACTORED Job Submission Endpoints ---

async def submit_job(
    analysis_type: str,
    files_dict: Dict[str, UploadFile], # <-- CHANGED: Now accepts a dict
    params: Dict[str, Any],
    task_queue: Any # Dependency
):
    """Helper function to enqueue any analysis job."""
    
    # 1. Create a unique ID for this analysis run
    job_uuid = str(uuid.uuid4())
    job_folder = UPLOAD_DIR / job_uuid
    job_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # 2. Save all 5 files using the new dict-based helper
        if len(files_dict) != 5:
            raise HTTPException(status_code=400, detail="Expected exactly 5 file fields.")
        
        saved_paths = await save_uploaded_files(files_dict, job_folder) # <-- Pass dict
        
        # 3. Enqueue the Master Job
        job = task_queue.enqueue(
            run_analysis_job,
            args=(
                job_uuid,
                analysis_type,
                saved_paths,
                params
            ),
            job_timeout='2h',
            result_ttl=86400, # Keep result in Redis for 1 day
            job_id=f"analysis-{job_uuid}" # Use a predictable RQ job ID
        )

        # 4. Return the RQ job ID for polling
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}

    except Exception as e:
        if job_folder.exists():
            shutil.rmtree(job_folder)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@api_router.post("/submit/static", summary="Submit a Static Reporters analysis")
async def submit_static_job(
    active_traj: UploadFile = File(...),
    active_topo: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    inactive_topo: UploadFile = File(...),
    config: UploadFile = File(...),
    task_queue: get_queue = Depends(),
):
    # CHANGED: Create a dict to preserve file identity
    files_dict = {
        "active_traj": active_traj, "active_topo": active_topo,
        "inactive_traj": inactive_traj, "inactive_topo": inactive_topo,
        "config": config
    }
    return await submit_job("static", files_dict, {}, task_queue)

@api_router.post("/submit/dynamic", summary="Submit a Dynamic (Transfer Entropy) analysis")
async def submit_dynamic_job(
    active_traj: UploadFile = File(...),
    active_topo: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    inactive_topo: UploadFile = File(...),
    config: UploadFile = File(...),
    te_lag: int = Form(10),
    task_queue: get_queue = Depends(),
):
    # CHANGED: Create a dict
    files_dict = {
        "active_traj": active_traj, "active_topo": active_topo,
        "inactive_traj": inactive_traj, "inactive_topo": inactive_topo,
        "config": config
    }
    params = {"te_lag": te_lag}
    return await submit_job("dynamic", files_dict, params, task_queue)

@api_router.post("/submit/qubo", summary="Submit a QUBO analysis")
async def submit_qubo_job(
    active_traj: UploadFile = File(...),
    active_topo: UploadFile = File(...),
    inactive_traj: UploadFile = File(...),
    inactive_topo: UploadFile = File(...),
    config: UploadFile = File(...),
    target_switch: str = Form(...),
    task_queue: get_queue = Depends(),
):
    # CHANGED: Create a dict
    files_dict = {
        "active_traj": active_traj, "active_topo": active_topo,
        "inactive_traj": inactive_traj, "inactive_topo": inactive_topo,
        "config": config
    }
    params = {"target_switch": target_switch}
    return await submit_job("qubo", files_dict, params, task_queue)

# --- Results Endpoints ---

@api_router.get("/results", summary="List all available analysis results")
async def get_results_list():
    # ... (existing get_results_list code remains unchanged)
    results_list = []
    try:
        for result_file in RESULTS_DIR.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                # Return just the metadata, not the full (large) result payload
                results_list.append({
                    "job_id": data.get("job_id"),
                    "analysis_type": data.get("analysis_type"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "error": data.get("error"),
                })
            except Exception:
                print(f"Failed to read result file: {result_file}")
        
        results_list.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return results_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {e}")

@api_router.get("/results/{job_uuid}", summary="Get the full JSON data for a specific result")
async def get_result_detail(job_uuid: str):
    # ... (existing get_result_detail code remains unchanged)
    try:
        result_file = RESULTS_DIR / f"{job_uuid}.json"
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"Result file for job '{job_uuid}' not found.")
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        return data
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to read result: {e}")