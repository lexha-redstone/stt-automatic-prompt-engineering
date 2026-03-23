import os
import re
import shutil
import urllib.parse
import subprocess

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_html = os.path.join(BASE_DIR, "evaluation_report.html")
    public_dir = os.path.join(BASE_DIR, "public")
    output_html = os.path.join(public_dir, "index.html")
    audio_dir = os.path.join(public_dir, "audio")

    if not os.path.exists(input_html):
        print(f"Error: '{input_html}' not found. Please generate the report first.")
        return

    os.makedirs(audio_dir, exist_ok=True)

    with open(input_html, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_audio_src(match):
        encoded_path = match.group(1)
        if not encoded_path:
            return match.group(0)
            
        original_path = urllib.parse.unquote(encoded_path)
        
        if os.path.exists(original_path) and os.path.isfile(original_path):
            filename = os.path.basename(original_path)
            dest_path = os.path.join(audio_dir, filename)
            
            try:
                if not os.path.exists(dest_path):
                    shutil.copy2(original_path, dest_path)
            except Exception as e:
                print(f"Error copying {original_path}: {e}")
                
            new_src = f"audio/{urllib.parse.quote(filename)}"
            return f'src="{new_src}"'
        else:
            return match.group(0)

    # The audio tag in 7_generate_report.py looks like:
    # src="{urllib.parse.quote(audio_path)}"
    new_content = re.sub(r'src="([^"]*)"', replace_audio_src, content)
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"✅ HTML updated and saved to {output_html}")
    print(f"✅ Audio files successfully copied to {audio_dir}")

    print("🚀 Deploying to Firebase...")
    try:
        # Run firebase deploy
        subprocess.run(["firebase", "deploy"], check=True, cwd=BASE_DIR)
        print("✅ Firebase deploy completed successfully.")
    except FileNotFoundError:
        print("❌ Firebase CLI not found. Please ensure it is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Firebase deploy failed with error code: {e.returncode}")

if __name__ == "__main__":
    main()
