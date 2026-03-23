import os
import sys
import json
import time
import glob
import subprocess
from google import genai
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# 0. Configs
# ==========================================
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
EXECUTION_LOG_DIR = os.path.join(BASE_DIR, "execution_log")

def run_script(script_name):
    """Executes individual Python files located in the src directory."""
    script_path = os.path.join(SRC_DIR, script_name)
    print(f"\n" + "="*50)
    print(f">>> Running: {script_name} ...")
    print("="*50)
    
    # The working directory is fixed to the project's top-level directory (BASE_DIR).
    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Failed to execute {script_name}. Terminating the pipeline.")
        sys.exit(result.returncode)
    
    print(f"\n[OK] {script_name} execution completed.")

def get_latest_log_file(timeout=5):
    """Locates the most recently created JSON log file in the execution_log folder."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        log_files = glob.glob(os.path.join(EXECUTION_LOG_DIR, "batch_jobs_log_*.json"))
        if log_files:
            latest_file = max(log_files, key=os.path.getctime)
            return latest_file
        time.sleep(1)
    return None

def extract_jobs_from_log(log_path):
    """Extracts the list of submitted jobs (job_name) from the JSON log file."""
    jobs = []
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    if "job_name" in item:
                        jobs.append(item["job_name"])
            except json.JSONDecodeError:
                pass
    return jobs

def poll_batch_jobs(job_names):
    """Waits for all submitted batch jobs to complete using the Vertex AI SDK."""
    if not job_names:
        print("No jobs to monitor. Proceeding to the next step...")
        return

    # Terminal states for Vertex AI jobs
    TERMINAL_STATES = ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PARTIALLY_SUCCEEDED"]
    
    print("\n" + "-"*50)
    print("--- Waiting for all Vertex AI Batch Jobs to complete ---")
    
    # GenAI Google Client Activation
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    active_jobs = list(job_names)
    start_time = time.time()
    
    while active_jobs:
        jobs_to_check = list(active_jobs)
        print("\n[Checking job status...]")
        for job_name in jobs_to_check:
            try:
                # Check status
                job = client.batches.get(name=job_name)
                state_str = str(job.state).upper()
                print(f" Job Name: {job_name} | State: {state_str}")
                
                # Remove from queue if terminal state
                if any(terminal in state_str for terminal in TERMINAL_STATES):
                    print(f" ✅ Job completed: {job_name}")
                    active_jobs.remove(job_name)
                    
            except Exception as e:
                print(f" ⚠️ Error checking status ({job_name}): {e}")
        
        # If there are still unfinished jobs, wait
        if active_jobs:
            elapsed = int(time.time() - start_time)
            print(f"\n=> Waiting for {len(active_jobs)} jobs to complete... (Elapsed: {elapsed} seconds)")
            print("=> Checking status again in 30 seconds...\n")
            time.sleep(30)
            
    print("\n--- All distributed batch processing jobs (Batch Jobs) have completed! ---")
    print("-"*50 + "\n")

def main():
    print("🚀 Starting the entire STT prompt optimization pipeline automation...")
    all_submitted_jobs = []

    # ---------------------------------------------------------
    # Step 1: 3-a General Data Inference
    # ---------------------------------------------------------
    run_script("3-a-batch-inference.py")
    log_file_3a = get_latest_log_file()
    all_submitted_jobs.extend(extract_jobs_from_log(log_file_3a))

    # Wait a moment to ensure clear ctime separation
    time.sleep(2) 

    # ---------------------------------------------------------
    # Step 2: 3-b Silence Data Inference
    # ---------------------------------------------------------
    run_script("3-b-silence-batch-inference.py")
    log_file_3b = get_latest_log_file()
    
    if log_file_3b != log_file_3a:
        all_submitted_jobs.extend(extract_jobs_from_log(log_file_3b))
        
    # Remove duplicate job numbers
    all_submitted_jobs = list(set(all_submitted_jobs)) 
    
    # ---------------------------------------------------------
    # Step 3: Polling for Asynchronous Processing Completion
    # ---------------------------------------------------------
    if all_submitted_jobs:
        print(f"\nFound a total of {len(all_submitted_jobs)} distributed batch jobs in the recent execution.")
        poll_batch_jobs(all_submitted_jobs)
    else:
        print("\nNo submitted jobs found in the recent execution history, proceeding to the next step without waiting.")

    # ---------------------------------------------------------
    # Step 4: Evaluate Performance
    # ---------------------------------------------------------
    run_script("4-evaluate-performance.py")

    # ---------------------------------------------------------
    # Step 5: Critic
    # ---------------------------------------------------------
    run_script("5-critic.py")

    # ---------------------------------------------------------
    # Step 6: Error-based Few-Shot Prompt Optimization
    # ---------------------------------------------------------
    run_script("6-few-shot-optimize.py")
    
    # ---------------------------------------------------------
    # Step 7: Final Status Report Generation (Optional)
    # ---------------------------------------------------------
    run_script("7_generate_report.py")

    print("\n🎉 The entire pipeline processing has been successfully completed!!")

if __name__ == "__main__":
    main()
