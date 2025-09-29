# Chat with LLM (HF Hub) 

> A minimal, production-ready Streamlit chat app for Hugging Face Inference API with streaming, multi-sample decoding, JSON-mode, and safe response sanitization. This README documents the code in `app.py`.

---

## ✨ Features

- **Streamlit UI** with chat bubbles + sidebar controls  
- **Hugging Face `InferenceClient`** for chat completions  
- **Token streaming** (with graceful fallback to non-streaming)  
- **Multi-sample decoding (`n`)** in non-streaming mode  
- **JSON schema** response formatting (toggle)  
- **Provider “extras”** via `extra_body` (e.g., `top_k`, `repetition_penalty`)  
- **Chat history truncation** for lower latency  
- **Response sanitization** (removes `<think>…</think>` blocks)  
- **System prompt editor** and one-click chat reset  
- **Env + secrets** support for `HUGGINGFACE_API_KEY`

---

## 🚀 Quickstart

### 1) Prerequisites
- Python 3.10+
- A Hugging Face API token with access to the chosen models

### 2) Clone & install
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_DIR>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, create one:
```txt
streamlit>=1.36
huggingface_hub>=0.24
python-dotenv>=1.0
langchain-core>=0.2
```

### 3) Provide your API key (pick one)
- **Environment variable**:
  ```bash
  export HUGGINGFACE_API_KEY=hf_xxx   # macOS/Linux
  set HUGGINGFACE_API_KEY=hf_xxx      # Windows (CMD)
  ```
- **Streamlit secrets** (great for Streamlit Cloud): create `.streamlit/secrets.toml`:
  ```toml
  HUGGINGFACE_API_KEY = "hf_xxx"
  ```

### 4) Run the app
```bash
streamlit run main.py
```

### 5) Open in the browser
Streamlit prints a local URL (usually `http://localhost:8501`). Open it and chat!

---

## 📦 How to use on your system (step-by-step)

1. **Create a virtual environment** and **install dependencies** (see Quickstart).  
2. **Set `HUGGINGFACE_API_KEY`** (env var or `secrets.toml`).  
3. **Run** `streamlit run app.py`.  
4. **Pick a model** in the sidebar and tweak decoding controls.  
5. **Type a message** at the bottom; the assistant streams or responds at once.  
6. (Optional) **Force JSON** responses, edit **system prompt**, or enable **provider extras**.  
7. (Optional) Click **🧹 Clear chat** to reset state instantly.

---

## 🧰 Sidebar controls — parameters (1-liners)

| Control | Description |
|---|---|
| **Model** | Select the HF Hub model used for chat completions (e.g., `mistralai/Mistral-7B-Instruct-v0.3`). |
| **temperature** | Randomness of generation; higher = more diverse outputs. |
| **top_p** | Nucleus sampling cutoff; sample from tokens within cumulative probability `p`. |
| **max_tokens** | Maximum tokens to generate in the response. |
| **presence_penalty** | Penalize tokens that already appeared (encourages new topics). |
| **frequency_penalty** | Penalize frequent tokens (reduces repetition). |
| **seed (0 = None)** | Deterministic sampling seed; `0` disables fixed seeding. |
| **n (samples)** | Number of completions to request (non-streaming only in this UI). |
| **stop sequences** | Comma-separated strings that, when generated, stop the output. |
| **Force JSON response_format** | Asks the model to return strict JSON conforming to a minimal schema. |
| **Provider extras → Enable** | Toggles sending `extra_body` to the backend (e.g., TGI-specific knobs). |
| **top_k** *(extras)* | Sample only from the top-k most likely tokens. |
| **repetition_penalty** *(extras)* | Penalizes token reuse to mitigate loops. |
| **Stream tokens** | Enable server-side streaming for realtime token updates. |
| **History: last N messages** | Truncate chat history to the last N (excl. system) to control context length. |
| **System prompt (editor)** | Global instruction for the assistant; apply to reset chat with new system prompt. |
| **🧹 Clear chat** | Resets messages/history and refreshes the session. |

---

## 🧪 Programmatic API wrapper — parameters (1-liners)

The app calls a thin wrapper around `InferenceClient.chat_completion` named `safe_chat_completion(...)`:

| Parameter | Description |
|---|---|
| **client** | A `huggingface_hub.InferenceClient` instance (already bound to a model + token). |
| **model** | Model repo ID on HF Hub (e.g., `mistralai/Mistral-7B-Instruct-v0.3`). |
| **messages** | OpenAI-style message list: `{"role": "user/assistant/system", "content": "…"}`. |
| **temperature** | Controls randomness; higher yields more varied text. |
| **max_tokens** | Upper bound on generated tokens. |
| **top_p** | Nucleus sampling; restricts to the smallest probability mass ≥ `p`. |
| **presence_penalty** | Encourages introducing new tokens not previously used. |
| **frequency_penalty** | Discourages repeating the same tokens often. |
| **seed** | Deterministic seed; `None` for non-deterministic sampling. |
| **stop** | List of stop strings to cut off generation early. |
| **response_format** | JSON schema instruction (when forcing JSON-compliant output). |
| **n** | Number of choices to return (non-streaming path supports multi-choice). |
| **stream** | If `True`, yields token chunks for realtime display. |
| **extra_body** | Provider-specific dict forwarded to backend (e.g., `{"top_k": 40, "repetition_penalty": 1.1}`). |

**Helpers explained (1-liners):**
- `get_api_key()` — Reads API key from Streamlit secrets or environment.  
- `sanitize_response(text)` — Strips `<think>…</think>` blocks and trims whitespace.  
- `extract_text_choices(resp)` — Normalizes HF object/dict responses to a list of strings.  
- `extract_usage(resp)` — Pulls token usage fields if the backend returns them.  
- `obj_or_dict(getter, fallback)` — Safely access fields from object- or dict-like responses.

---

## 🧱 How it works (at a glance)

```
[User Input] → Streamlit chat_input
            → Build OpenAI-style messages (system + recent history)
            → safe_chat_completion(...) via InferenceClient
              ├─ stream=True: yield chunks → live UI update
              └─ stream=False: collect choices → render (with multi-sample expanders)
            → sanitize_response() removes <think> blocks
            → Save to session_state + LangChain InMemoryChatMessageHistory
```

---

## 🔧 Configuration notes

- **API key resolution order**: Streamlit `secrets` → `HUGGINGFACE_API_KEY` env var.  
- **Streaming + `n > 1`**: the UI **forces** `n=1` when streaming (documented in-app).  
- **History truncation**: `History: last N messages` lets you cap the prompt length.  
- **JSON mode**: When enabled, a minimal schema is sent to nudge valid JSON objects.  
- **Provider extras**: Sent under `extra_body` only when **Enable** is checked.

---

## 🧪 Local development tips

- Add print/log points in `safe_chat_completion()` to inspect backend payloads.  
- Extend the **Model** dropdown to include your org’s private models.  
- Adjust `max_history_msgs` default to trade off latency vs. context depth.  
- Wrap `sanitize_response()` logic if you use different “reasoning” tags.

---

## 🔐 Security

- **Never hardcode** tokens; prefer secrets or environment variables.  
- Treat streamed content as **untrusted**; sanitization removes hidden tags.  
- If deploying on Streamlit Cloud, store `HUGGINGFACE_API_KEY` in **Secrets**.

---

## 🧯 Troubleshooting

- **“No API key found”**: Set `HUGGINGFACE_API_KEY` (env or secrets) and reload.  
- **Model errors**: Ensure your token has access and the repo ID is correct.  
- **Weird JSON**: Disable “Force JSON” or relax your downstream parser.  
- **Duplicate/loopy output**: Raise `repetition_penalty` or `frequency_penalty`.  
- **Latency**: Lower `max_tokens`, reduce history length, or disable streaming.

---

## 🙌 Acknowledgments

- Built with **Streamlit** and **Hugging Face Inference API**.  
- Optional LangChain memory primitives for future extensibility.

## Contact - 
- yashdshah2403@gmail.com
- https://www.linkedin.com/in/shah-yash24/
