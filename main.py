# app.py
import os
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from huggingface_hub import InferenceClient

# Optional: keep if you plan to use LangChain history elsewhere
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)
# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Chat with LLM (HF Hub)", page_icon="💬", layout="centered")


# ----------------------------
# Helpers
# ----------------------------
def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets (Streamlit Cloud), fallback to env var
    key = None
    try:
        key = st.secrets.get("HUGGINGFACE_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        pass
    return key or os.getenv("HUGGINGFACE_API_KEY")


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def sanitize_response(text: str) -> str:
    """Remove hidden 'reasoning' blocks and stray whitespace."""
    if not text:
        return text
    # Remove entire <think>...</think> blocks
    text = THINK_TAG_RE.sub("", text)
    return text.strip()


def obj_or_dict(getter, fallback):
    """
    Try attribute-based access (pydantic/dataclass), otherwise dict-style.
    `getter` should be a lambda that does attribute access.
    `fallback` should be a lambda that does dict access.
    """
    try:
        return getter()
    except Exception:
        return fallback()


def safe_chat_completion(
    client: InferenceClient,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    n: int = 1,
    stream: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
):
    """
    Wrapper that handles both streaming and non-streaming, and object/dict HF response variants.
    """
    try:
        if stream:
            # Streaming yields ChatCompletionChunk-like events
            return client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
                stop=stop,
                response_format=response_format,
                n=1,  # streaming multiple choices is messy; force n=1 when streaming
                stream=True,
                extra_body=extra_body,
            )
        # Non-streaming
        resp = client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            stop=stop,
            response_format=response_format,
            n=n,
            stream=False,
            extra_body=extra_body,
        )
        return resp
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")


def extract_text_choices(resp: Any) -> List[str]:
    """Return list of assistant message texts from a non-streaming response."""
    def from_obj():
        return [c.message.content or "" for c in resp.choices]

    def from_dict():
        return [c["message"]["content"] or "" for c in resp["choices"]]

    choices = obj_or_dict(from_obj, from_dict)
    return [sanitize_response(x) for x in choices]


def extract_usage(resp: Any) -> Optional[Dict[str, int]]:
    """Return token usage if present."""
    try:
        # object-like
        u = getattr(resp, "usage", None)
        if u:
            return {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
    except Exception:
        pass
    try:
        # dict-like
        u = resp.get("usage")
        if u:
            return {
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "total_tokens": u.get("total_tokens"),
            }
    except Exception:
        pass
    return None


# ----------------------------
# API key
# ----------------------------
api_key = get_api_key()
if not api_key:
    st.error("No API key found. Set `HUGGINGFACE_API_KEY` in **st.secrets** or environment.")
    st.stop()


# ----------------------------
# Sidebar: settings
# ----------------------------
st.sidebar.header("⚙️ Settings")

# Model & core decoding
model_selection = st.sidebar.selectbox(
    "Model",
    [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-2b-it",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ],
    index=0,
)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.7, 0.01)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 1.0, 0.01)
max_tokens = st.sidebar.number_input("max_tokens", min_value=16, max_value=8192, value=512, step=16)

# Penalties & sampling count
presence_penalty = st.sidebar.slider("presence_penalty", -2.0, 2.0, 0.0, 0.1)
frequency_penalty = st.sidebar.slider("frequency_penalty", -2.0, 2.0, 0.0, 0.1)
seed = st.sidebar.number_input("seed (0 = None)", min_value=0, value=0, step=1)
n_choices = st.sidebar.number_input("n (samples)", min_value=1, max_value=5, value=1, step=1)

# Stop & JSON response format
stop_str = st.sidebar.text_input("stop sequences (comma-separated)", value="")
use_json = st.sidebar.checkbox("Force JSON response_format", value=False)

# Provider-specific extras (e.g., TGI)
with st.sidebar.expander("Provider extras (via extra_body)"):
    use_extras = st.checkbox("Enable", value=False)
    top_k = st.slider("top_k", 1, 200, 40, 1, disabled=not use_extras)
    repetition_penalty = st.slider("repetition_penalty", 0.8, 2.0, 1.1, 0.01, disabled=not use_extras)

# UX controls
streaming = st.sidebar.toggle("Stream tokens", value=True)
max_history_msgs = st.sidebar.slider("History: last N messages (excl. system)", 2, 200, 60, 2)

st.sidebar.markdown("---")
with st.sidebar.expander("System prompt"):
    default_sys = "You are a helpful AI assistant."
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = default_sys
    system_prompt_new = st.text_area("Prompt", value=st.session_state.system_prompt, height=120)
    col_sp1, col_sp2 = st.columns(2)
    with col_sp1:
        if st.button("Apply (reset chat)", use_container_width=True):
            st.session_state.system_prompt = system_prompt_new.strip() or default_sys
            # Reset chat to only system
            st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]
            st.session_state.chat_history = InMemoryChatMessageHistory()
            st.rerun()
    with col_sp2:
        if st.button("Reset to default", use_container_width=True):
            st.session_state.system_prompt = default_sys
            st.session_state.messages = [{"role": "system", "content": default_sys}]
            st.session_state.chat_history = InMemoryChatMessageHistory()
            st.rerun()

col1, col2 = st.sidebar.columns(2)
with col1:
    clear_clicked = st.button("🧹 Clear chat", use_container_width=True)
with col2:
    st.caption("")  # spacing


# ----------------------------
# Client (cache the HTTP session)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_client(model: str, token: str):
    return InferenceClient(model, token=token)

client = get_client(model_selection, api_key)


# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = [
        {"role": "system", "content": st.session_state.get("system_prompt", "You are a helpful AI assistant.")}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

if clear_clicked:
    st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.rerun()


# ----------------------------
# App body
# ----------------------------
st.title("Chat with LLM")

# Render previous messages (skip system)
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            st.markdown(m["content"].replace("\n", "  \n"))
        else:
            st.text(m["content"])

# Input
user_input = st.chat_input("Enter your message...")

if user_input:
    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.add_message(HumanMessage(content=user_input))

    # Show it immediately
    with st.chat_message("user"):
        st.text(user_input)

    # Build request payload
    stops = [s.strip() for s in stop_str.split(",") if s.strip()] or None
    extras = {"top_k": top_k, "repetition_penalty": repetition_penalty} if use_extras else None

    resp_fmt = None
    if use_json:
        # Minimal JSON schema that forces valid JSON object with "answer" string
        resp_fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": True,
                },
                "strict": True,
            },
        }

    # Truncate context to reduce latency/cost
    sys_msg = st.session_state.messages[0:1]
    tail = st.session_state.messages[-max_history_msgs:]
    messages_for_api = sys_msg + tail

    # If streaming + n>1, force n=1 (documented to the user)
    effective_n = n_choices
    if streaming and n_choices > 1:
        effective_n = 1
        st.info("Streaming multiple samples is not supported here; forcing n=1 for this call.")

    # Call the API
    with st.chat_message("assistant"):
        try:
            if streaming:
                placeholder = st.empty()
                acc = ""
                with st.status("Thinking...", expanded=False):
                    events = safe_chat_completion(
                        client=client,
                        model=model_selection,
                        messages=messages_for_api,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        seed=None if seed == 0 else seed,
                        stop=stops,
                        response_format=resp_fmt,
                        n=effective_n,
                        stream=True,
                        extra_body=extras,
                    )
                    for event in events:
                        # Handle both object and dict chunk deltas
                        try:
                            delta = event.choices[0].delta.content or ""
                        except Exception:
                            delta = event["choices"][0]["delta"].get("content", "") or ""
                        if delta:
                            acc += delta
                            placeholder.markdown(sanitize_response(acc).replace("\n", "  \n"))
                assistant_text = sanitize_response(acc)
                st.markdown(assistant_text.replace("\n", "  \n"))
                # Persist the assistant message to continue the thread
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                st.session_state.chat_history.add_message(AIMessage(content=assistant_text))
            else:
                with st.status("Thinking...", expanded=False):
                    resp = safe_chat_completion(
                        client=client,
                        model=model_selection,
                        messages=messages_for_api,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        seed=None if seed == 0 else seed,
                        stop=stops,
                        response_format=resp_fmt,
                        n=effective_n,
                        stream=False,
                        extra_body=extras,
                    )
                # Extract assistant text(s)
                choices_text = extract_text_choices(resp)

                if effective_n == 1:
                    assistant_text = choices_text[0] if choices_text else ""
                    st.markdown(assistant_text.replace("\n", "  \n"))
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                    st.session_state.chat_history.add_message(AIMessage(content=assistant_text))
                else:
                    # Show each candidate in its own expander
                    for i, txt in enumerate(choices_text, start=1):
                        with st.expander(f"Choice {i}"):
                            st.markdown(txt.replace("\n", "  \n"))
                    # Continue the thread with the first choice by default
                    if choices_text:
                        st.session_state.messages.append({"role": "assistant", "content": choices_text[0]})
                        st.session_state.chat_history.add_message(AIMessage(content=choices_text[0]))

                # Optional usage display (if backend provides it)
                usage = extract_usage(resp)
                if usage:
                    st.caption(
                        f"Tokens — prompt: {usage.get('prompt_tokens')}, "
                        f"completion: {usage.get('completion_tokens')}, "
                        f"total: {usage.get('total_tokens')}"
                    )

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
