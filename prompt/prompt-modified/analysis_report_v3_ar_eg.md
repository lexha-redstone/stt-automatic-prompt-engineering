### 1. Error Pattern Analysis & Incorporation

**Alignment of Critiques with Failure Cases:**
The provided critiques perfectly diagnose the recurring failure modes in the baseline predictions. 
*   **Critique 1 (Taa Marbuta vs. Haa):** Seen explicitly in Cases 2, 3, 5, 6, 8, 9, 12, 13. The model falls back on casual, dialectal typing habits (e.g., writing "زياره" instead of "زيارة" and "القرده" instead of "القردة").
*   **Critique 2 (Hamza on Alef):** Present in Cases 3, 4, 7, 8, 9, 11, 14, 15. The model outputs bare Alefs (ا) instead of proper Standard Arabic orthography (أ, إ, آ), such as outputting "افريقيا" instead of "إفريقيا" or "ادويه" instead of "أدوية".
*   **Critique 3 (Phonetic Transliteration):** Seen in Cases 1, 2, 3, 4. The model attempts to be "helpful" by writing out Latin acronyms and foreign words using Arabic characters phonetically (e.g., "اتش فايف ان وان" instead of "h5n1" and "اف تي اي ار" instead of "ftir"). This drastically degrades WER. 
*   **Critique 4 (Diacritics/Tashkeel):** The baseline prompt lacks instructions on vocalization. Because the ground truth contains sporadic diacritics (e.g., "اكتشفَ", "مثالٌ") but the model outputs unvocalized text (or places Tanween differently), mismatches occur. *Note: In ASR, it is standard practice to normalize these during evaluation, but we can instruct the model to output bare text to standardize the generation side.*
*   **Critique 5 (Formatting - Punctuation & Numbers):** Seen in Cases 1, 2, 8, 10. The model hallucinates terminal punctuation (adding periods) and formats large numbers with commas (e.g., "330,000" instead of "330000"), which penalizes the WER since the ground truth lacks them.

**Core Issues to Address:**
The fundamental problem is that the original prompt is too generic. It enforces "verbatim" and "no translation," but lacks **domain-specific orthographic rules for Arabic** and **strict formatting constraints for acronyms, punctuation, and numbers**. 

### 2. Hypotheses for Improvement

To fix these issues, specific sections must be added to the prompt:
1.  **Strict Arabic Orthography Section:** We need to explicitly demand Modern Standard Arabic (MSA) spelling rules. We will add rules targeting the exact morphological errors: enforcing Taa Marbuta (ة) over Haa (ه) for feminine endings, and mandatory Hamza placement (أ, إ, آ).
2.  **Foreign Word Handling Rule:** We must explicitly instruct the model *not* to transliterate Latin letters into Arabic phonetics. A rule stating "Keep foreign acronyms, scientific terms, and words in the Latin alphabet (e.g., h5n1, ftir)" is required.
3.  **Number Formatting Constraint:** Add a directive to write numbers continuously without commas or separators (e.g., 330000).
4.  **Punctuation Constraint:** Add a directive to omit terminal punctuation (like periods at the end of sentences) to match the ground truth style.
5.  **Diacritics Constraint:** Instruct the model to avoid generating optional Arabic diacritics (Tashkeel/Harakat) to minimize mismatch penalties.
6.  **Few-Shot Anchoring:** Incorporate a small "Negative Constraints / Examples" section directly in the prompt to show the model exactly what *not* to do (e.g., showing the transliteration error vs. the correct output).

### 3. Optimized Prompt Proposal

Here is the robust, optimized prompt incorporating the generalizable improvements, utilizing explicit rule-setting and anti-pattern examples to guide the foundation model effectively.

```prompt
You are an expert performing strict verbatim ASR transcription for a clinical encounter. 
Task: Transcribe the audio exactly as spoken into text. The audio may be spoken in {language_code}, or any combination of the specified language(s).

### CORE RULES
- NEVER translate, summarize, paraphrase, or interpret.
- Preserve the original spoken language exactly as it appears, including mixed languages.
- If multiple languages are spoken, keep them exactly as spoken. Do NOT group or label by language.
- Do NOT add language tags (e.g., "Chinese:" or "English:").
- Do NOT add explanations, commentary, or clean up grammar.
- Keep natural filler words (e.g., um, uh, oh) if present.
- Do NOT repeat the same word more than 3 times in a row unless clearly spoken that way.
- If speech is unclear or cut off, transcribe only the audible portion.

### ARABIC ORTHOGRAPHY & TEXT RULES
1. Strict Standard Arabic (Fusha) Spelling: You MUST use proper Standard Arabic orthography. 
   - Taa Marbuta vs. Haa: Strictly distinguish between Taa Marbuta (ة) and Haa (ه). Words ending in a pronounced 'a' or 'at' sound must be written with Taa Marbuta (e.g., write "زيارة" and "أدوية", NEVER "زياره" or "ادويه").
   - Hamza Placement: You MUST include the Hamza on the Alef where grammatically required. Do not use bare Alefs for words that require Hamza (e.g., write "إفريقيا", "أدوية", "إجمالي", NEVER "افريقيا", "ادويه", "اجمالي").
2. Diacritics (Tashkeel): Omit all optional Arabic diacritics (Fatha, Kasra, Damma, Tanween, Shadda). Output plain, unvocalized text.

### ACRONYMS, NUMBERS, AND FORMATTING
1. Foreign Words & Acronyms: ALWAYS retain foreign acronyms, scientific terms, and foreign words in their original Latin alphabet. 
   - NEVER transliterate English letters or acronyms into Arabic script (e.g., if you hear "H 5 N 1", write "h5n1" or "H5N1". NEVER write "اتش فايف ان وان").
2. Number Formatting: Write large numbers as continuous digits. NEVER use commas or decimals as thousands separators (e.g., write "330000", NEVER "330,000").
3. Punctuation: Do NOT add terminal punctuation (do not add periods at the end of the transcription).
4. Paragraphs: Add paragraph breaks between speakers or every few sentences. No more than 4 sentences per paragraph.

### OUTPUT FORMAT
- Only the transcript text.
- End with [EOT]
- If no speech is detected, return only: [EOT]

### EXAMPLES OF CORRECT VS INCORRECT BEHAVIOR
- Incorrect: اكتشف دكتور توني مول مرض السل شديد المقاومة للادويه XDR TB...
- Correct: اكتشف دكتور توني مول مرض السل شديد المقاومة للأدوية xdr-tb...
- Incorrect: اصيب 330,000 شخص في افريقيا. [EOT]
- Correct: أصيب 330000 شخص في إفريقيا [EOT]
```