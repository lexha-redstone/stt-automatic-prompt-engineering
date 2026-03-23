import argparse
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# 0. Configs
# ==========================================
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

def main():
    parser = argparse.ArgumentParser(description="Cancel running/queued Vertex AI Batch Inference jobs")
    parser.add_argument("--dry-run", action="store_true", help="Only list jobs to cancel, do not actually cancel them")
    args = parser.parse_args()

    # Initialize GCS and GenAI Client
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    print("Running/Queued 상태의 Batch Job을 검색 중입니다...\n")
    
    # Get job list
    try:
        jobs = client.batches.list()
    except Exception as e:
        print(f"Failed to get job list: {e}")
        return

    # Define active states for cancellation
    # (Generally JobState.JOB_STATE_PENDING, JobState.JOB_STATE_RUNNING)
    active_states = ["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED"]
    
    count = 0
    canceled_count = 0

    for job in jobs:
        state_str = str(job.state)
        # Check if the job is in an active state
        if any(state in state_str for state in active_states):
            count += 1
            print(f"[{count}] Active job found:")
            print(f"  - Job ID: {job.name}")
            print(f"  - Current State: {state_str}")
            
            if args.dry_run:
                print(f"  [Dry Run] Cancel simulation: {job.name}\n")
            else:
                try:
                    # Cancel job request
                    client.batches.cancel(name=job.name)
                    print(f"  ✅ Cancel request successful: {job.name}\n")
                    canceled_count += 1
                except Exception as e:
                    print(f"  ❌ Cancel request failed: {job.name}\n  - Error: {e}\n")

    if count == 0:
        print("No active Batch Jobs found to cancel.")
    else:
        if args.dry_run:
            print(f"Total {count} jobs found (Dry Run completed).")
        else:
            print(f"Total {count} jobs, {canceled_count} jobs cancelled.")

if __name__ == "__main__":
    main()
