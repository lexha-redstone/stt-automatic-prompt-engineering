import csv
import json
import os
import argparse
import jiwer
import urllib.parse

# ==========================================
# 0. Configs
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "execution_log", "evaluation_summary.csv")
FALLBACK_CSV_PATH = os.path.join(BASE_DIR, "archive", "execution_log_archived", "evaluation_summary.csv")
OUTPUT_HTML = os.path.join(BASE_DIR, "evaluation_report.html")
BATCH_RESULTS_TEMP_DIR = os.path.join(BASE_DIR, "batch_results")
PROMPT_DIR_MAIN = os.path.join(BASE_DIR, "prompt")
PROMPT_DIR_MODIFIED = os.path.join(PROMPT_DIR_MAIN, "prompt-modified")

# Optional alias for backward compatibility in the script
SCRIPT_DIR = BASE_DIR

def generate_report():
    actual_csv_path = CSV_PATH
    if not os.path.exists(actual_csv_path):
        if os.path.exists(FALLBACK_CSV_PATH):
            actual_csv_path = FALLBACK_CSV_PATH
        else:
            print(f"Error: {CSV_PATH} not found.")
            return

    data = []
    with open(actual_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse metrics
            try:
                row['wer'] = float(row['wer'])
            except:
                row['wer'] = 0.0
            try:
                row['cer'] = float(row['cer'])
            except:
                row['cer'] = 0.0
            data.append(row)

    # Sort data by lang, group, model, config, prompt_version
    data.sort(key=lambda x: (x.get('lang', ''), x.get('group', ''), x.get('model', ''), x.get('config', ''), x.get('prompt_version', '')))

    # Group data by group and lang for dataset tab
    groups = {}
    for row in data:
        k = (row['group'], row['lang'])
        if k not in groups:
            groups[k] = []
        groups[k].append(row)

    dataset_html_sections = []
    for (group, lang), items in sorted(groups.items()):
        if not items:
            continue
        best_item = min(items, key=lambda x: x['cer'])
        csv_path = best_item.get('local_inference_csv_path', '')
        
        if not os.path.exists(csv_path) and csv_path:
            basename = os.path.basename(csv_path)
            alt_path = os.path.join(BATCH_RESULTS_TEMP_DIR, basename)
            if os.path.exists(alt_path):
                csv_path = alt_path

        dataset_html_sections.append(f"<h3>{lang} ({group}) - Best: {best_item['model']} (Prompt: {best_item.get('prompt_version', '')}, Config: {best_item.get('config', '')})</h3>")
        dataset_html_sections.append(f"<p><b>CSV:</b> {csv_path} | <b>CER:</b> {best_item['cer']*100:.2f}%</p>")
        
        if os.path.exists(csv_path):
            dataset_html_sections.append("<table><tr><th>ID</th><th>Audio</th><th>Transcription</th><th>Prediction</th><th>WER</th><th>CER</th></tr>")
            
            try:
                with open(csv_path, "r", encoding="utf-8-sig") as bf:
                    breader = csv.DictReader(bf)
                    
                    transformation = jiwer.Compose([
                        jiwer.ToLowerCase(),
                        jiwer.RemoveMultipleSpaces(),
                        jiwer.RemovePunctuation(),
                        jiwer.Strip(),
                    ])

                    count = 0
                    for brow in breader:
                        if count >= 20: # Reduced limit to improve page load speed
                            break
                        count += 1
                        
                        ref = str(brow.get('transcription', ''))
                        hyp = str(brow.get('prediction', ''))
                        hyp = hyp.replace('[EOT]', '').strip()
                        
                        if not ref.strip(): continue
                        
                        r_trans = transformation(ref)
                        h_trans = transformation(hyp)
                        
                        if not r_trans.strip(): continue
                        if not h_trans.strip(): h_trans = " "
                        
                        try:
                            w = jiwer.wer(r_trans, h_trans)
                            c = jiwer.cer(r_trans, h_trans)
                        except:
                            w = 0.0
                            c = 0.0
                        
                        audio_path = brow.get('noisy_output_path', '')
                        if not audio_path:
                            audio_path = brow.get('silence_output_path', '')
                        if not audio_path:
                            audio_path = brow.get('audio', '')
                        
                        audio_tag = f'<audio controls preload="none" src="{urllib.parse.quote(audio_path)}">Your browser does not support the audio element.</audio>'
                        
                        dataset_html_sections.append(f"<tr><td>{brow.get('id', '')}</td><td>{audio_tag}</td><td>{ref}</td><td>{hyp}</td><td>{w*100:.2f}%</td><td>{c*100:.2f}%</td></tr>")
            except Exception as e:
                dataset_html_sections.append(f"<tr><td colspan='6'>Error reading CSV: {e}</td></tr>")
            dataset_html_sections.append("</table><br>")
        else:
            dataset_html_sections.append(f"<p>CSV not found: {csv_path}</p><br>")

    dataset_html = "\n".join(dataset_html_sections)

    prompts_html_sections = []
    prompt_dirs = [PROMPT_DIR_MAIN, PROMPT_DIR_MODIFIED]
    for p_dir in prompt_dirs:
        if os.path.exists(p_dir):
            for filename in sorted(os.listdir(p_dir)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(p_dir, filename)
                    display_name = os.path.relpath(filepath, os.path.join(SCRIPT_DIR, "prompt"))
                    try:
                        with open(filepath, "r", encoding="utf-8") as pf:
                            content = pf.read()
                        content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        prompts_html_sections.append(f"<h3>{display_name}</h3><pre style='background:#fff; padding:15px; border-radius:8px; border:1px solid #ddd; overflow-x:auto; white-space: pre-wrap; font-family: monospace;'>{content}</pre>")
                    except Exception as e:
                        prompts_html_sections.append(f"<h3>{display_name}</h3><p>Error reading file: {e}</p>")
    prompts_html = "\n".join(prompts_html_sections)

    # Find best model per language based on CER (excluding train group, only test/silence)
    best_per_lang = {}
    for row in data:
        if row.get('group') not in ['test', 'silence']: continue
        lang = row.get('lang')
        if not lang: continue
        if lang not in best_per_lang or row['cer'] < best_per_lang[lang]['cer']:
            best_per_lang[lang] = row

    best_per_lang_sections = []
    for lang in sorted(best_per_lang.keys()):
        best = best_per_lang[lang]
        best_per_lang_sections.append(f"<p style=\"text-indent: 2em;\"><b>{lang}</b>: {best['model']} | Prompt: {best.get('prompt_version') or 'default'} | Config: {best.get('config') or 'default'} (CER: {best['cer']*100:.2f}%)</p>")
    best_per_lang_html = "\n                ".join(best_per_lang_sections)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HH : STT Automatic Prompt Optimization Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f7fa; }}
        h1 {{ text-align: center; color: #333; }}
        .tabs {{ display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; justify-content: center; }}
        .tab {{ padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; font-weight: bold; color: #555; }}
        .tab.active {{ color: #2980b9; border-bottom: 3px solid #2980b9; }}
        .tab-content {{ display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .tab-content.active {{ display: block; }}
        .controls {{ display: flex; gap: 20px; margin-bottom: 20px; justify-content: center; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        select {{ padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #2980b9; color: white; cursor: pointer; position: sticky; top: 0; }}
        th:hover {{ background-color: #1abc9c; }}
        .graph-container {{ width: 100%; max-width: 1000px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .filters {{ font-weight: bold; }}
        .executive-summary {{ 
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%); 
            padding: 30px; 
            border-radius: 15px; 
            border-left: 10px solid #2980b9; 
            margin-bottom: 40px; 
            box-shadow: 0 10px 25px rgba(41, 128, 185, 0.1);
            border-top: 1px solid #e1effe;
            border-right: 1px solid #e1effe;
            border-bottom: 1px solid #e1effe;
        }}
        .executive-summary h2 {{ 
            margin-top: 0; 
            color: #2c3e50; 
            font-size: 28px;
            border-bottom: 3px solid #2980b9;
            padding-bottom: 12px;
            margin-bottom: 25px;
            display: inline-block;
        }}
        .executive-summary p {{
            color: #34495e;
            font-size: 16px;
        }}
        .executive-summary b {{
            color: #2980b9;
        }}
        .dataset-info {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 20px; }}
        audio {{ width: 100%; min-width: 200px; }}
    </style>
</head>
<body>
    <h1>HH : STT Automatic Prompt Optimization Report</h1>
    
    <div class="tabs">
        <button class="tab active" onclick="openTab('summary')">Executive Summary</button>
        <button class="tab" onclick="openTab('results')">Results</button>
        <button class="tab" onclick="openTab('dataset')">Dataset</button>
        <button class="tab" onclick="openTab('prompts')">Prompts</button>
    </div>

    <div id="summary" class="tab-content active">
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report was prepared to measure how the Speech-to-Text (STT) performance of Gemini models varies based on different prompts and configurations.
                <br><br>
                1. <b>Evaluation Dataset</b> : The evaluation dataset was constructed by <b>filtering the google/fleurs data for medical and healthcare-related content</b>, with random noise added to simulate real-world conditions.<br>
                <p style="text-indent: 2em;"> 1-a. <b>9</b> Languages : ["en_us", "cmn_hans_cn", "yue_hant_hk", "ko_kr", "hi_in", "ja_jp", "ar_eg", "he_il", "el_gr"] = [English, Mandarin, Cantonese, Korean, Hindi, Japanese, Arabic, Hebrew, Greek] </p>
                <p style="text-indent: 2em;"> 1-b. (Type #1) Train & Test : audio with random noise </p>
                <p style="text-indent: 2em;"> 1-c. (Type #2) Silence : audio with 10s silence inserted front / middle / last of it. </p>
                <p style="text-indent: 2em;"> 1-d. To prevent overfitting, the data used for optimization was partitioned into a train set, while a distinct, independent test set was utilized for inference & evaluation.</p>
                <p style="text-indent: 2em;"> 1-e. Please note that the v3 prompts are optimized for this synthetically generated data and have not been tailored to your production data.</p>

                2. <b>Prompt optimization</b> : Optimizing the prompt up to 3-4 times is effective as it increases alingmnet with the specific dataset. However, further optimization tends to cause overfitting, failing to imporve overall STT performance.
                <p style="text-indent: 2em;"> 2-a. The iteration process : Inference (Batch API) → Evaluate (Calculate WER/CER) → Error Analysis (Select Worst cases & Build hypothesis) → Use Meta-prompt to Optimize → Inference</p>
                <p style="text-indent: 2em;"> 2-b. Meta prompt used : <code>meta-prompt.txt</code> found in the Prompts tab. </p>
                <p style="text-indent: 2em;"> 2-c. Critic-based approach : An LLM acting as a Critic reviews the WER/CER for each row and passes only generalizable knowledge to the optimizer. </p>
                
                3. <b>Configurations</b>
                <p style="text-indent: 2em;"> 3-a. default</p>
                <p style="text-indent: 2em;"> 3-b. temperature: "audio_timestamp": True, "temperature": 0.2</p>
                <p style="text-indent: 2em;"> 3-c. audio_ts_true: "audio_timestamp": True, "topK": 1, "topP": 0.1 </p>

                4. <b>Addressing Hallucination Issues</b> : Reducing hallucinations is positively correlated with improving STT performance. As seen in the Results tab, the v3 prompts demonstrated superior performance, particularly with silence data.

                <p style="text-indent: 2em;">4-a. Token Management: Setting the <code>max_output_tokens</code> value to <b>n</b> times the input audio length would prevent excessive token usage.</p>
                <p style="text-indent: 4em;">-> Total token usage w/ 3x faster speech for 1 second : <b>267</b> (3.1-fl) / <b>419</b> (3-flash) / <b>316</b> (2.5-pro) </p>
                
                5. <b>Best performing models for each language</b>
                {best_per_lang_html}
            </p>
        </div>
    </div>

    <div id="results" class="tab-content">
        <div id="bestSettingContainer" style="text-align: center; margin-bottom: 20px; font-size: 1.2em; font-weight: bold; color: #2980b9;"></div>
        
        <div class="controls">
            <label class="filters">
                Language:
                <select id="langFilter" onchange="updateView()"></select>
            </label>
            <label class="filters">
                Group:
                <select id="groupFilter" onchange="updateView()"></select>
            </label>
        </div>

        <div class="graph-container">
            <canvas id="metricsChart"></canvas>
        </div>

        <table id="dataTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Language &#x21D5;</th>
                    <th onclick="sortTable(1)">Group &#x21D5;</th>
                    <th onclick="sortTable(2)">Model &#x21D5;</th>
                    <th onclick="sortTable(3)">Config &#x21D5;</th>
                    <th onclick="sortTable(4)">Prompt Version &#x21D5;</th>
                    <th onclick="sortTable(5)">WER (%) &#x21D5;</th>
                    <th onclick="sortTable(6)">CER (%) &#x21D5;</th>
                    <th onclick="sortTable(7)">CSV Path &#x21D5;</th>
                </tr>
            </thead>
            <tbody id="tableBody">
            </tbody>
        </table>
    </div>

    <div id="dataset" class="tab-content">
        <h2>Details (Audio Transcript Comparison)</h2>
        <div class="dataset-info">
            <h3>Dataset</h3>
            <p>The dataset used for this evaluation consists of <b>google/fleurs</b> data combined with random noise & silence to simulate real-world noisy environments. <br>Train data : each 50 rows for 9 languages <br> Test data : each 50 rows for 9 languages <br> Silence data : each 30 rows for 9 languages</p>
        </div>
        <br>
        {dataset_html}
    </div>

    <div id="prompts" class="tab-content">
        <h2>Prompts Used</h2>
        <div class="dataset-info">
            <p>These are the prompts found in the <code>prompt/</code> directory.</p>
        </div>
        {prompts_html}
    </div>

    <script>
        const rawData = {json.dumps(data)};
        let chartInstance = null;

        function openTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }}

        function init() {{
            const langs = [...new Set(rawData.map(d => d.lang))].sort();
            const groups = [...new Set(rawData.map(d => d.group))].sort();

            const langSelect = document.getElementById('langFilter');
            const groupSelect = document.getElementById('groupFilter');

            langSelect.innerHTML = '<option value="all">All</option>' + langs.map(l => `<option value="${{l}}">${{l}}</option>`).join('');
            groupSelect.innerHTML = '<option value="all">All</option>' + groups.map(g => `<option value="${{g}}">${{g}}</option>`).join('');

            if (langs.includes('en_us')) langSelect.value = 'en_us';
            if (groups.includes('test')) groupSelect.value = 'test';

            updateView();
        }}

        function updateView() {{
            const selectedLang = document.getElementById('langFilter').value;
            const selectedGroup = document.getElementById('groupFilter').value;

            const filteredData = rawData.filter(d => 
                (selectedLang === 'all' || d.lang === selectedLang) &&
                (selectedGroup === 'all' || d.group === selectedGroup)
            );

            if (filteredData.length > 0) {{
                let best = filteredData.reduce((prev, current) => (prev.cer < current.cer) ? prev : current);
                document.getElementById('bestSettingContainer').innerText = `Best Setting (by CER) = Model: ${{best.model}} | Prompt: ${{best.prompt_version || 'default'}} | Config: ${{best.config}} (CER: ${{(best.cer * 100).toFixed(2)}}%)`;
            }} else {{
                document.getElementById('bestSettingContainer').innerText = '';
            }}

            renderTable(filteredData);
            renderChart(filteredData);
        }}

        function renderTable(data) {{
            if (data.length === 0) {{
                document.getElementById('tableBody').innerHTML = '';
                return;
            }}
            
            const minWer = Math.min(...data.map(d => d.wer));
            const minCer = Math.min(...data.map(d => d.cer));

            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = data.map(d => {{
                const isBestWer = d.wer === minWer;
                const isBestCer = d.cer === minCer;
                const werStyle = isBestWer ? 'background-color: #d4edda; color: #155724; font-weight: bold;' : '';
                const cerStyle = isBestCer ? 'background-color: #d4edda; color: #155724; font-weight: bold;' : '';
                
                return `
                <tr>
                    <td>${{d.lang}}</td>
                    <td>${{d.group}}</td>
                    <td><strong>${{d.model}}</strong></td>
                    <td><em>${{d.config}}</em></td>
                    <td>${{d.prompt_version || ''}}</td>
                    <td style="${{werStyle}}">${{(d.wer * 100).toFixed(2)}}%</td>
                    <td style="${{cerStyle}}">${{(d.cer * 100).toFixed(2)}}%</td>
                    <td style="font-size: 0.85em; color: #555; word-break: break-all;">${{d.local_inference_csv_path}}</td>
                </tr>
            `}}).join('');
        }}

        function renderChart(data) {{
            const labels = [];
            const werData = [];
            const cerData = [];

            const grouped = {{}};
            data.forEach(d => {{
                let key = `${{d.model}} (${{d.config}}) [${{d.prompt_version || 'default'}}]`;
                if (document.getElementById('langFilter').value === 'all') key += ` [${{d.lang}}]`;
                if (document.getElementById('groupFilter').value === 'all') key += ` [${{d.group}}]`;

                if (!grouped[key]) {{
                    grouped[key] = {{ wer: [], cer: [] }};
                }}
                grouped[key].wer.push(d.wer * 100);
                grouped[key].cer.push(d.cer * 100);
            }});

            for (const [key, metrics] of Object.entries(grouped)) {{
                labels.push(key);
                const avgWer = metrics.wer.reduce((a, b) => a + b, 0) / metrics.wer.length;
                const avgCer = metrics.cer.reduce((a, b) => a + b, 0) / metrics.cer.length;
                
                werData.push(parseFloat(avgWer.toFixed(2)));
                cerData.push(parseFloat(avgCer.toFixed(2)));
            }}

            const minWer = Math.min(...werData.filter(v => !isNaN(v)));
            const minCer = Math.min(...cerData.filter(v => !isNaN(v)));

            const werBgColors = werData.map(val => val === minWer ? 'rgba(40, 167, 69, 0.8)' : 'rgba(54, 162, 235, 0.7)');
            const werBorderColors = werData.map(val => val === minWer ? 'rgba(40, 167, 69, 1)' : 'rgba(54, 162, 235, 1)');
            
            const cerBgColors = cerData.map(val => val === minCer ? 'rgba(40, 167, 69, 0.8)' : 'rgba(255, 99, 132, 0.7)');
            const cerBorderColors = cerData.map(val => val === minCer ? 'rgba(40, 167, 69, 1)' : 'rgba(255, 99, 132, 1)');

            const ctx = document.getElementById('metricsChart').getContext('2d');
            
            if (chartInstance) {{
                chartInstance.destroy();
            }}

            chartInstance = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'WER (%)', data: werData, backgroundColor: werBgColors, borderColor: werBorderColors, borderWidth: 1 }},
                        {{ label: 'CER (%)', data: cerData, backgroundColor: cerBgColors, borderColor: cerBorderColors, borderWidth: 1 }}
                    ]
                }},
                options: {{ 
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Average WER and CER Comparison', font: {{ size: 18 }} }} }},
                    scales: {{ 
                        y: {{ beginAtZero: true }},
                        x: {{ ticks: {{ display: labels.length < 30, maxRotation: 45, minRotation: 45 }} }} 
                    }}
                }}
            }});
        }}

        function sortTable(n) {{
            var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            table = document.getElementById("dataTable");
            switching = true;
            dir = "asc"; 
            while (switching) {{
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {{
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[n];
                    y = rows[i + 1].getElementsByTagName("TD")[n];
                    let xVal = x.innerHTML.replace('%', '');
                    let yVal = y.innerHTML.replace('%', '');
                    if (!isNaN(parseFloat(xVal))) {{ xVal = parseFloat(xVal); yVal = parseFloat(yVal); }}
                    else {{ xVal = xVal.toLowerCase(); yVal = yVal.toLowerCase(); }}
                    
                    if (dir == "asc") {{
                        if (xVal > yVal) {{ shouldSwitch = true; break; }}
                    }} else if (dir == "desc") {{
                        if (xVal < yVal) {{ shouldSwitch = true; break; }}
                    }}
                }}
                if (shouldSwitch) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount ++; 
                }} else {{
                    if (switchcount == 0 && dir == "asc") {{
                        dir = "desc";
                        switching = true;
                    }}
                }}
            }}
        }}

        window.onload = init;
    </script>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report successfully generated at {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_report()
