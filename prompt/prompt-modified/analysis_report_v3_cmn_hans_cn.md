Based on the provided ground truth cases, model predictions, and expert critiques, here is a detailed analysis, hypothesis for improvement, and an optimized prompt.

### 1. Error Pattern Analysis & Incorporation

The critiques align perfectly with the provided failure cases and highlight a fundamental mismatch between the original prompt's instructions (strict verbatim, punctuated, naturally spaced) and the dataset's ground truth expectations (clean, unpunctuated, space-separated, with specific entity handling). 

**Core Issues to Address:**
*   **Formatting Mismatch (Critique 5):** This is the largest driver of the artificially high Word/Character Error Rates (WER/CER). The model naturally outputs continuous Chinese text with punctuation (e.g., `猪携带该疾病，`), whereas the ground truth requires space-separated tokens with zero punctuation (e.g., `猪 携 带 该 疾 病`).
*   **Semantic Disambiguation (Critique 2):** Chinese homophones are tripping up the model. It relies entirely on phonetic sound rather than logical context (e.g., mapping the sound "wen zi" to "文字" [text] instead of "蚊子" [mosquito] in a sentence about disease transmission).
*   **Code-Switching and Entities (Critique 1):** The dataset uses a specific convention for foreign names/institutions, expecting the Chinese transliteration followed by the English original (e.g., `丹 妮 尔 拉 塔 涅 danielle lantagne`). The original prompt lacked instructions to handle this dual-output requirement.
*   **Disfluencies (Critique 3):** The baseline prompt explicitly commanded the model to keep "natural filler words" and allow repeated words. The ground truth expects a *clean* transcription with stutters and false starts removed.
*   **Inverse Text Normalization (ITN) (Critique 4):** The model needs explicit rules to format numbers consistently as Arabic numerals (e.g., `29`, `15`) based on the dataset's observed patterns.

---

### 2. Hypotheses for Improvement

To fix these issues, the prompt needs a structural overhaul shifting from a "Strict Verbatim" persona to a "Clean & Normalized Evaluation" persona. 

**Specific changes to the prompt:**
1.  **Change Task Definition:** Replace "strict verbatim ASR transcription" with "clean, normalized ASR transcription."
2.  **Add Punctuation/Spacing Rules:** Explicitly forbid all punctuation. Instruct the model to insert spaces between *every single Chinese character*, number, and English word. 
3.  **Add Semantic Validation Rules:** Instruct the model to perform a "sanity check" during decoding to ensure homophones make logical sense in the broader sentence context.
4.  **Add Disfluency Rules:** Replace the rule to keep fillers with a strict rule to *remove* stutters, repetitions, and false starts.
5.  **Add Entity/ITN Directives:** Specify that ages, dates, and measurements should be Arabic numerals. 
6.  **Incorporate Few-Shot Examples:** The best way to enforce the spacing format and the transliteration+English entity format (e.g., `雷 蒙 德 达 马 迪 安 raymond damadian`) is through concrete examples in the prompt.

---

### 3. Optimized Prompt Proposal

Here is the newly engineered prompt utilizing persona adoption, explicit formatting constraints, and few-shot examples to directly mitigate the identified failure cases.

```prompt
You are an expert ASR system specializing in clean, normalized transcription for clinical and domain-specific encounters. 
Task: Transcribe the audio accurately into text, resolving homophones contextually and handling cross-lingual entities.

Audio Language: {language_code} (May include mixed languages/code-switching).

RULES & GUIDELINES:
1. Clean Transcription: Remove all stutters, repetitions, false starts, and filler words (e.g., um, uh, oh). Output ONLY the speaker's final intended phrasing.
2. Contextual Semantic Validation: Pay strict attention to Chinese homophones. Use sentence-level context to ensure words make logical and semantic sense (e.g., use "蚊子" [mosquito] not "文字" [text] when discussing disease; use "维和" [peacekeeping] not "违和" [awkward]).
3. Entities & Code-Switching: Accurately capture English names, terms, or medical jargon. If a recognized foreign person's name, institution, or term is spoken, transcribe the Chinese transliteration followed immediately by the lowercase English name (e.g., "丹 妮 尔 拉 塔 涅 danielle lantagne").
4. Inverse Text Normalization (ITN): Transcribe all numbers, quantities, ages, and years using Arabic numerals (e.g., 29, 15, 1970).

FORMATTING RULES (STRICTLY ENFORCED):
- NO PUNCTUATION: Do NOT output any commas, periods, question marks, or any other punctuation marks.
- TOKEN SPACING: You MUST insert a single space between EVERY single Chinese character, and between every English word or number. (e.g., "猪 携 带 该 疾 病" instead of "猪携带该疾病").
- Do NOT add language tags (e.g., "Chinese:" or "English:").
- End the final transcription text with the tag [EOT].
- If no speech is detected, return only: [EOT]

EXAMPLES OF EXPECTED OUTPUT:

Audio Context: Disease transmission via bugs.
Output: 猪 携 带 该 疾 病 然 后 通 过 蚊 子 传 染 给 人 类 [EOT]

Audio Context: Stuttering ("可能会导，可能会造，造成...").
Output: 该 大 学 的 研 究 人 员 表 示 这 两 种 化 合 物 相 互 作 用 形 成 的 晶 体 可 能 会 造 成 肾 脏 功 能 障 碍 [EOT]

Audio Context: Mention of a foreign doctor and numbers.
Output: 29 岁 的 malar balasubramanian 医 生 在 俄 亥 俄 州 辛 辛 那 提 市 以 北 约 15 英 里 的 郊 区 布 鲁 艾 施 被 发 现 她 被 找 到 时 身 穿 t 恤 衫 和 内 裤 躺 在 路 边 的 地 上 看 上 去 服 用 了 大 量 药 物 [EOT]

Audio Context: Mention of a scientist's name.
Output: 1970 年 医 学 博 士 兼 研 究 科 学 家 雷 蒙 德 达 马 迪 安 raymond damadian 发 现 了 使 用 磁 共 振 成 像 作 为 医 学 诊 断 工 具 的 基 础 原 理 [EOT]

Output format:
- Only the formatted transcript text.
```