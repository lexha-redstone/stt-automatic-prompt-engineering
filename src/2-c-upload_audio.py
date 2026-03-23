import pandas as pd
import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 0. Configs
# ==========================================
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

NOISY_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_processed.csv")
NOISY_OUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_processed_with_gcs.csv")

SILENCE_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_silence_processed.csv")
SILENCE_OUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_silence_processed_with_gcs.csv")

def upload_and_save_csv(csv_path, audio_path_col, gcs_audio_dir_prefix, bucket, output_csv_path):
    if not os.path.exists(csv_path):
        print(f"Skipped: {csv_path} file not exists")
        return

    print(f"\n[{csv_path}] Processing...")
    df = pd.read_csv(csv_path)
    
    gcs_uris = []

    for index, row in df.iterrows():
        audio_path = row.get(audio_path_col, row.get("file_path"))
        if not audio_path or pd.isna(audio_path):
            gcs_uris.append(None)
            continue
            
        filename = os.path.basename(audio_path)
        lang = row.get('lang', 'unknown')
        group = row.get('group', 'unknown')
        
        gcs_audio_dir = f"{gcs_audio_dir_prefix}/{group}_{lang}"
        gcs_blob_path = f"{gcs_audio_dir}/{filename}"

        blob = bucket.blob(gcs_blob_path)
        gcs_uri = f"gs://{BUCKET_NAME}/{gcs_blob_path}"
        
        if not blob.exists():
            upload_src = audio_path
            if os.path.exists(upload_src):
                print(f"    - Uploading audio: {filename} -> {gcs_uri}")
                blob.upload_from_filename(upload_src)
                gcs_uris.append(gcs_uri)
            else:
                print(f"      Warning: Local audio file not found ({upload_src})")
                gcs_uris.append(None)
        else:
            print(f"    - Already exists: {filename}")
            gcs_uris.append(gcs_uri)

    df['gcs_uri'] = gcs_uris
    df.to_csv(output_csv_path, index=False)
    print(f"Result saved to: {output_csv_path}")

def main():
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)

    upload_and_save_csv(NOISY_CSV_PATH, "noisy_output_path", "batch_audio", bucket, NOISY_OUT_CSV_PATH)
    
    upload_and_save_csv(SILENCE_CSV_PATH, "silence_output_path", "batch_audio_silence", bucket, SILENCE_OUT_CSV_PATH)

if __name__ == "__main__":
    main()
