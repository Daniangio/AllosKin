import os
import shutil
import yaml
import json
import traceback
from pathlib import Path
from datetime import datetime
from rq import get_current_job
from typing import Dict, Any, Optional
from alloskin.pipeline.runner import run_analysis

# Define the persistent results directory
RESULTS_DIR = Path("/app/data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Master Analysis Job ---

def run_analysis_job(
    job_uuid: str,
    analysis_type: str, 
    file_paths: Dict[str, str],
    params: Dict[str, Any],
    config_path: Optional[str] = None,
    residue_selections_dict: Optional[Dict[str, str]] = None
):
    """
    The main, long-running analysis function.
    This function is executed by the RQ Worker and handles all analysis types.
    """
    job = get_current_job()
    start_time = datetime.utcnow()
    
    # This ID is needed by the frontend to link from the results page
    # back to the live status page.
    rq_job_id = job.id if job else f"analysis-{job_uuid}" # Reconstruct as fallback
    
    result_payload = {
        "job_id": job_uuid,
        "rq_job_id": rq_job_id, # <-- NEW: Store the RQ ID
        "analysis_type": analysis_type,
        "status": "started",
        "created_at": start_time.isoformat(),
        "params": params,
        "residue_selections_mapping": None,
        "results": None,
        "error": None,
        "completed_at": None, # Will be filled in 'finally'
    }
    
    result_filepath = RESULTS_DIR / f"{job_uuid}.json"

    def save_progress(status_msg: str, progress: int):
        if job:
            job.meta['status'] = status_msg
            job.meta['progress'] = progress
            job.save_meta()
        print(f"[Job {job_uuid}] {status_msg}")

    def write_result_to_disk(payload: Dict[str, Any]):
        """Helper to write the result payload to the persistent JSON file."""
        try:
            with open(result_filepath, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"Saved result to {result_filepath}")
        except Exception as e:
            print(f"CRITICAL: Failed to save result file {result_filepath}: {e}")
            # If saving fails, update the in-memory payload for the final RQ return
            payload["status"] = "failed"
            payload["error"] = f"Failed to save result file: {e}"

    try:
        save_progress("Initializing...", 0)
        write_result_to_disk(result_payload)

        # Step 1: Load residue selections if a config file is provided
        residue_selections = residue_selections_dict
        if config_path and not residue_selections:
            print(f"[Worker] Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            residue_selections = config.get('residue_selections')

        # Step 2: Prepare arguments for the runner
        # The runner expects specific keys for file paths
        runner_file_paths = {
            'active_traj': file_paths['active_traj_file'],
            'active_topo': file_paths['active_topo_file'],
            'inactive_traj': file_paths['inactive_traj_file'],
            'inactive_topo': file_paths['inactive_topo_file'],
        }
        
        # Step 3: Delegate to the core runner
        job_results, mapping = run_analysis(
            analysis_type=analysis_type,
            file_paths=runner_file_paths,
            params=params,
            residue_selections=residue_selections,
            progress_callback=save_progress
        )
        result_payload["residue_selections_mapping"] = mapping

        # Step 4: Finalize
        result_payload["status"] = "finished"
        result_payload["results"] = job_results
        
    except Exception as e:
        print(f"[Job {job_uuid}] FAILED: {e}")
        traceback.print_exc()
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        # --- Re-raise the exception AFTER saving the result. ---
        # This ensures that the RQ job itself is marked as 'failed',
        # which is what the frontend status page is polling for.
        raise e
    
    finally:
        # Step 5: Save final persistent JSON result
        save_progress("Saving final result", 95)
        result_payload["completed_at"] = datetime.utcnow().isoformat()
        
        write_result_to_disk(result_payload)

        # Clean up the temporary upload folder
        try:
            upload_dir = Path("/app/data/uploads") / job_uuid
            if upload_dir.exists() and upload_dir.name == job_uuid:
                shutil.rmtree(upload_dir)
                print(f"Cleaned up upload directory: {upload_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up upload dir: {e}")

    # This is the value returned to RQ and shown on the status page
    return result_payload