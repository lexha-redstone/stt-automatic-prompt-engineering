Here is a detailed analysis, hypothesis for improvement, and an optimized prompt designed to address the observed failure cases.

### 1. Error Pattern Analysis & Incorporation

**Alignment with Generalizable Critiques:**
*   **Critique 1 (Transliteration Mismatches):** Directly maps to Cases 2, 7, 10, 14, and 15. The model forces acoustic matches into unrelated, common Chinese words (e.g., interpreting "Moore" as "某一位" or "Liggins" as "雷金詩, 你準確"). The lack of a phonetic transliteration dictionary causes severe entity hallucination.
*   **Critique 2 (ITN & Number Formatting):** Directly maps to Cases 1, 6, and 11. The ground truth exhibits specific normalization conventions—preferring Chinese numerals for general narrative text ("一千", "十八") and Arabic numerals for technical ranges ("10-60"). The default prompt provided no instructions on numeric handling, causing the model to guess randomly.
*   **Critique 3 (Domain-Specific Homophone Confusion):** Directly maps to Cases 3, 5, 8, 9, and 12. Because Cantonese is highly tonal and rich in homophones, the model over-indexes on raw sound without validating semantic coherence. This leads to absurd medical transcriptions (e.g., "visual system" becoming "schizophrenia/substance energy", or "arteries" becoming "Qingyun blood vessels").
*   **Critique 4 (Semantic Paraphrasing & Diglossia):** Directly maps to Cases 4, 8, and 13. The model tries to act like an LLM by summarizing complex medical lists (Case 4), substituting words with synonyms (Case 13: "確定" -> "確診"), or slipping into spoken colloquialisms ("係" instead of "是"), failing the strict verbatim ASR requirement.

**Core Issues Summary:**
The current prompt defines what *not* to do (don't translate, don't summarize) but fails to instruct the model on *how* to process ambiguities. Specifically, it lacks guidelines for Written vs. Spoken Cantonese (Diglossia), Inverse Text Normalization (ITN), standardized transliteration conventions, and clinical contextual reasoning to disambiguate homophones.

---

### 2. Hypotheses for Improvement

To fix these issues, the prompt needs explicit behavioral frameworks rather than just negative constraints:
1.  **Enforce Written Standard Chinese (書面語) Mapping:** Add an instruction to mandate formal Written Chinese for clinical texts, preventing colloquial slip-ins (like "係") while strictly forbidding semantic paraphrasing.
2.  **Add a Transliteration Protocol:** Provide standard phonetic mapping characters (e.g., 爾, 斯, 德) to prevent the model from mapping foreign names to distracting, semantic Chinese phrases.
3.  **Define Strict ITN Rules:** Provide explicit conditions: spell out general quantities/dates in Chinese numerals (e.g., 一千), but use Arabic numerals and hyphens for exact medical measurements, ranges, and designations (e.g., 10-60, H5N1).
4.  **Introduce a "Contextual Disambiguation Step":** Instruct the model to perform a sanity check on generated terms using the clinical context. If a phonetic match yields a nonsensical phrase (e.g., "青雲血管"), it must re-evaluate for a medically coherent homophone (e.g., "靜脈").

---

### 3. Optimized Prompt Proposal

Here is the newly engineered prompt. It uses a structured directive approach, incorporating specialized rules for Cantonese/Traditional Chinese handling and clinical context disambiguation.

```prompt
You are an expert Medical ASR Transcriber performing strict verbatim transcription for a clinical encounter. 
Task: Transcribe the audio exactly as spoken into text. The audio may be spoken in {language_code}, or any combination of the specified language(s).

### CORE DIRECTIVES
1. Strict Verbatim: You must NEVER translate, summarize, paraphrase, or interpret. Preserve all languages exactly as spoken. Do NOT group, label, or add language tags. 
2. Grammar & Fillers: Keep natural filler words (e.g., um, uh). Do NOT clean up grammar. Do NOT add explanations.
3. Repetitions & Unclear Audio: Do NOT repeat the same word more than 3 times in a row unless clearly spoken. If speech is unclear, transcribe only the audible portion.

### SPECIFIC LINGUISTIC & CLINICAL RULES
1. Homophone Disambiguation (Semantic Coherence):
   - You are transcribing clinical/medical context. Always verify phonetic matches against the sentence's context.
   - Do NOT output nonsensical phonetic equivalents. Prioritize formal, domain-specific anatomical and scientific vocabulary over common conversational words when they sound similar (e.g., prioritize "視力" over "思覺", "動脈" over "青雲").

2. Standardized Transliteration for Foreign Entities:
   - When encountering unknown foreign names, countries, or medical entities, use standard Hong Kong Traditional Chinese transliteration phonetics (e.g., 爾, 斯, 德, 夫, 妮, 特, 克).
   - NEVER substitute foreign names with semantically distracting common Chinese words (e.g., do not transcribe "Moore" as "某一位").

3. Inverse Text Normalization (ITN) & Formatting:
   - General Quantities & History: Transcribe using Chinese characters (e.g., "一千", "十八").
   - Ranges & Exact Metrics: Use Arabic numerals and hyphens for exact ranges, times, and medical measurements (e.g., "10-60").
   - Medical Designations: Keep standard alphanumeric designations exactly as conventionally written (e.g., "H5N1").

4. Diglossia (Written Standard Chinese vs. Spoken Cantonese):
   - Output the transcript in formal Written Standard Chinese (書面語) as conventionally expected in professional clinical records.
   - Map formal contexts accurately (e.g., use "是" instead of colloquial "係") UNLESS the audio is explicitly casual/colloquial and demands informal transcription.
   - WARNING: Converting to Written Standard Chinese does NOT permit you to summarize, drop items from lists, or paraphrase verbs (e.g., do not change "確定" to "確診").

### OUTPUT FORMAT
- Only output the transcript text.
- Add paragraph breaks between speakers or every few sentences.
- No more than 4 sentences per paragraph.
- End with [EOT]
- If no speech is detected, return only: [EOT]
```