import streamlit as st
import time
from langchain_core.messages import HumanMessage
from chatbot import app  # your compiled graph
import re



# --- Page Setup ---
st.set_page_config(page_title="Deep Scrape Chatbot", page_icon="🤖", layout="wide")

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
        .user-msg {
            background: linear-gradient(135deg, #E9D5FF, #C084FC);
            color: white;
            padding: 12px 16px;
            border-radius: 16px;
            margin: 8px 0;
            text-align: right;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            font-size: 1rem;
            line-height: 1.4;
        }
        .bot-msg {
            background: #F3E8FF;
            color: #2D2D2D;
            padding: 12px 16px;
            border-radius: 16px;
            margin: 8px 0;
            text-align: left;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            font-size: 1rem;
            line-height: 1.4;
        }
        .system-msg {
            color: #888;
            font-size: 0.85rem;
            margin: 6px 0;
            text-align: center;
            font-style: italic;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title / Branding ---
st.title("🤖 Ask Anything")
st.markdown("Your AI assistant that combines smart reasoning with deep web research.")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Helper: render assistant safely ---
def render_assistant(content: str):
    # Case 1: fenced code block
    code_block = re.search(r"```(\w+)?\n([\s\S]+?)```", content)
    if code_block:
        lang = code_block.group(1) or "text"
        code_text = code_block.group(2)
        st.code(code_text, language=lang)
    # Case 2: plain C++-like content without fences
    elif content.strip().startswith("#include") or "int main" in content:
        st.code(content, language="cpp")
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {content}</div>", unsafe_allow_html=True)

# --- Chat History ---
for role, content in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{content}</div>", unsafe_allow_html=True)
    elif role == "assistant":
        render_assistant(content)
    else:
        st.markdown(f"<div class='system-msg'>{content}</div>", unsafe_allow_html=True)

# --- Input Box ---
if prompt := st.chat_input("Type your question here..."):
    st.session_state["messages"].append(("user", prompt))
    st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("🤖 Thinking..."):
        final_response = ""
        searching_shown = False
        placeholder = st.empty()

        for event in app.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": "st-user"}}
        ):
            if "chatbot" in event:
                msg = event["chatbot"]["messages"][-1]

                if getattr(msg, "tool_calls", None) and not searching_shown:
                    st.toast("🔍 Searching the web...")
                    searching_shown = True

                elif getattr(msg, "content", None):
                    final_response = msg.content

                    # typing animation only for text (not code)
                    if not ("```" in final_response or "#include" in final_response):
                        typed_text = ""
                        for word in final_response.split():
                            typed_text += word + " "
                            placeholder.markdown(
                                f"<div class='bot-msg'>🤖 {typed_text.strip()}</div>",
                                unsafe_allow_html=True,
                            )
                            time.sleep(0.04)
                    else:
                        placeholder.empty()
                        render_assistant(final_response)

        st.session_state["messages"].append(("assistant", final_response))