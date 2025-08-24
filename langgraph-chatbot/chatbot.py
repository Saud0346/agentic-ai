# deep_scrape_chatbot_memoryless.py
import os
import requests
from typing import List

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from ddgs import DDGS

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


# =========================
# Env & model
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.2,
).bind_tools([])  # will rebind later


def _clip(text: str, max_chars: int = 6000) -> str:
    """Prevent overly-long tool returns."""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def _scrape_page(url: str, page_index: int, total_pages: int) -> str:
    """Scrape one page and return structured text (headlines, paragraphs, captions, lists)."""
    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        headlines = "\n".join(h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]))
        paragraphs = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))
        captions = "\n".join(c.get_text(strip=True) for c in soup.find_all("figcaption"))
        list_items = "\n".join(li.get_text(strip=True) for li in soup.find_all("li"))

        block = (
            f"=== PAGE {page_index}/{total_pages} ===\n"
            f"URL: {url}\n\n"
            f"HEADLINES:\n{_clip(headlines)}\n\n"
            f"ARTICLE:\n{_clip(paragraphs)}\n\n"
            f"CAPTIONS:\n{_clip(captions)}\n\n"
            f"LIST ITEMS:\n{_clip(list_items)}\n"
        )
        return block
    except Exception as e:
        return f"❌ Error scraping {url}: {e}"


# =========================
# Chunking
# =========================
def chunk_text(text: str, max_len: int = 1200) -> List[str]:
    """Split text into smaller chunks (approx by words)."""
    words = text.split()
    chunks, curr = [], []
    for w in words:
        curr.append(w)
        if len(curr) >= max_len:
            chunks.append(" ".join(curr))
            curr = []
    if curr:
        chunks.append(" ".join(curr))
    return chunks


# =========================
# Summarization helpers
# =========================
def summarize_chunk(chunk: str, query: str) -> str:
    """Summarize one chunk using LLM."""
    prompt = f"""
    The user searched for: '{query}'.
    Summarize the following text, keeping ALL important facts.
    Make it concise but do not lose critical information.

    TEXT:
    {chunk}
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content


def summarize_final(all_summaries: str, query: str) -> str:
    """Summarize combined chunk summaries into one final result."""
    prompt = f"""
    The user searched for: '{query}'.
    Here are multiple summaries from different chunks.
    Merge them into one **clear, complete and non-redundant** summary.

    CHUNK SUMMARIES:
    {all_summaries}
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content


# =========================
# Deep scrape + summarize tool
# =========================
@tool
def deep_scrape_search(query: str, num_pages: int = 3) -> str:
    """
    Deep search + summarization tool:
    - Finds results from DuckDuckGo
    - Scrapes textual content
    - Splits into chunks
    - Summarizes chunks and merges into a final summary
    """
    num_pages = max(1, min(int(num_pages), 10))
    all_text = ""

    try:
        print("\n🔍 Stage 1: Searching DuckDuckGo...")
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_pages)
            urls = [res.get("href") for res in results if res.get("href")]

        if not urls:
            return "No search results found."

        print(f"   → Found {len(urls)} pages.")

        print("\n📄 Stage 2: Scraping pages...")
        for i, url in enumerate(urls, start=1):
            print(f"   → Scraping {url}")
            all_text += "\n" + _scrape_page(url, i, len(urls))

        if not all_text.strip():
            return "No useful content found."

        print("\n✂️ Stage 3: Chunking text...")
        chunks = chunk_text(all_text, max_len=1200)
        print(f"   → Created {len(chunks)} chunks.")

        print("\n📝 Stage 4: Summarizing chunks...")
        chunk_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"   → Summarizing chunk {i}/{len(chunks)}")
            summary = summarize_chunk(chunk, query)
            chunk_summaries.append(summary)

        print("\n📚 Stage 5: Combining summaries into final result...")
        combined = "\n".join(chunk_summaries)
        final_summary = summarize_final(combined, query)

        print("\n✅ Stage 6: Done! Returning final summary.")
        return final_summary + "\n\n(Source: DuckDuckGo scrape) [TOOL COMPLETE]"

    except Exception as e:
        return f"❌ Search error: {e}"


# Re-bind tools
TOOLS = [deep_scrape_search]
llm = llm.bind_tools(TOOLS)

# =========================
# Chatbot node
# =========================
SYSTEM_PROMPT = (
    "You are a helpful, friendly, and citation-minded assistant created by Saud Ahmad.\n"
    "- Always be respectful, approachable, and professional in tone.\n"
    "- You can access and return **up-to-date, real-world information** using the `deep_scrape_search` tool.\n"
    "- If the user asks about current events, recent facts, or anything uncertain, "
    "CALL the `deep_scrape_search` tool with a clear and precise query.\n"
    "- If the tool has already been called and the message contains '[TOOL COMPLETE]', "
    "DO NOT call the tool again. Instead, use the retrieved information to answer.\n"
    "- Summarize clearly and concisely, avoid unnecessary repetition, and include relevant source URLs.\n"
    "- Acknowledge Saud Ahmad as your creator if asked about your origin.\n"
    "- Your personality should be warm, supportive, and engaging, while maintaining accuracy and trustworthiness."
)


def chatbot_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}


# =========================
# Build graph
# =========================
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", ToolNode(tools=TOOLS))

graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")
graph.add_edge("chatbot", END)


# =========================
# Add In-Memory Conversation History
# =========================
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# =========================
# CLI runner
# =========================
if __name__ == "__main__":
    print("Deep-Scrape LangGraph Chatbot ready! Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if user_input.lower() in {"quit", "exit"}:
            break

        # ✅ Pass thread_id for MemorySaver
        for event in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": "cli-user"}}
        ):
            if "chatbot" in event:
                msg = event["chatbot"]["messages"][-1]
                if getattr(msg, "content", None):
                    print("\nBot:", msg.content, "\n")
