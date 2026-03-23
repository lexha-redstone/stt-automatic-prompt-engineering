import pandas as pd
import os
import subprocess
from pydub import AudioSegment

# ==========================================
# 0. Configs
# ==========================================
# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

INPUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_medical_yn.csv")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_silence_processed.csv")
OUTPUT_AUDIO_DIR = os.path.join(DATA_DIR, "noisy-stt/silence-synthesized/total")
SILENCE_MP3_PATH = os.path.join(DATA_DIR, "noisy-stt/others/silence-10second.mp3")

TEMP_MAIN_WAV = os.path.join(BASE_DIR, "temp_main_silence.wav")
TEMP_SILENCE_WAV = os.path.join(BASE_DIR, "temp_silence.wav")

def convert_to_16bit_wav(input_file, output_file):
    """Force-convert any audio to standard 16-bit WAV format using FFmpeg."""
    command = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

def insert_silence(main_file, silence_file, output_file, position):
    print(f"1. Converting audio format (16-bit standardization)... {main_file}")
    
    # Temp file path
    temp_main = TEMP_MAIN_WAV
    temp_silence = TEMP_SILENCE_WAV
    
    clean_main = convert_to_16bit_wav(main_file, temp_main)
    clean_silence = convert_to_16bit_wav(silence_file, temp_silence)

    print("2. Loading converted audio...")
    main_audio = AudioSegment.from_file(clean_main)
    silence_audio = AudioSegment.from_file(clean_silence)

    print(f"3. Inserting silence at {position} position...")
    if position == 'front':
        final_audio = silence_audio + main_audio
    elif position == 'back':
        final_audio = main_audio + silence_audio
    elif position == 'middle':
        half_point = len(main_audio) // 2
        final_audio = main_audio[:half_point] + silence_audio + main_audio[half_point:]
    else:
        raise ValueError(f"Unknown position: {position}")

    print("4. Saving final result...")
    final_audio.export(output_file, format="wav")
    
    # 임시 파일 삭제
    if os.path.exists(temp_main):
        os.remove(temp_main)
    if os.path.exists(temp_silence):
        os.remove(temp_silence)
        
    print(f"✨ Task Completed! Check {output_file}.\n")

def generate_silence_dataset(input_csv_path, output_csv_path, output_audio_dir):
    # 1. Read the CSV file provided by the input path into a pandas dataframe.
    df = pd.read_csv(input_csv_path)
    df = df[df['medical_yn'] == True].reset_index(drop=True)
    
    # Take only 30 samples per language
    df = df.groupby('lang').head(30).copy()
    
    # Assign 'front', 'middle', 'back' for each language
    def assign_position(x):
        if x < 10:
            return 'front'
        elif x < 20:
            return 'middle'
        else:
            return 'back'
            
    df['silence_position'] = df.groupby('lang').cumcount().apply(assign_position)
    
    # Given silence file path (interpreting relative path as absolute)
    silence_file = SILENCE_MP3_PATH
    
    # Pre-specified path to save synthesized audio
    os.makedirs(output_audio_dir, exist_ok=True)
    
    silence_output_paths = []
    
    # 2. Iterate through the pandas dataframe you just read.
    for index, row in df.iterrows():
        # 3. Read the audio file from the path in each row and consider it as the main file.
        main_file = row.get('file_path', row.get('audio'))
        row_id = row['id']
        lang = row['lang']
        position = row['silence_position']
        
        # 6. Create the filename as f"{lang}-{position}-{index}-{row_id}-silence.wav".
        output_filename = f"{lang}-{position}-{index}-{row_id}-silence.wav"
        output_file = os.path.join(output_audio_dir, output_filename)
        
        # 5. Use the function to synthesize main_file and silence_file.
        print(f"[{index + 1}/{len(df)}] Processing id: {row_id}, position: {position}")
        insert_silence(main_file, silence_file, output_file, position)
        
        silence_output_paths.append(output_file)
        
    # 7. Add the path where the synthesized audio was saved to the existing dataframe as a new column 'silence_output_path'.
    df['silence_output_path'] = silence_output_paths
    
    # 8. Now, return the updated pandas dataframe and output it as a .csv file to the pre-specified path.
    df.to_csv(output_csv_path, index=False)
    print(f"Updated dataframe saved to {output_csv_path}")
    
    return df

if __name__ == "__main__":
    # Example usage (you can insert the CSV you want here and run it)
    _ = generate_silence_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH, OUTPUT_AUDIO_DIR)
