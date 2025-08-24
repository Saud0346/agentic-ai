# deep_scrape_chatbot_memoryless.py
import os
import requests
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from ddgs import DDGS

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
)

# =========================
# Scraping function
# =========================
def scrape_page(url: str) -> str:
    """Scrape textual content from one webpage"""
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        headlines = " ".join([h.get_text(" ", strip=True) for h in soup.find_all(["h1","h2","h3"])])
        paragraphs = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
        list_items = " ".join([li.get_text(" ", strip=True) for li in soup.find_all("li")])
        return f"{headlines}\n{paragraphs}\n{list_items}"

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return ""

# =========================
# Chunking
# =========================
def chunk_text(text: str, max_len: int = 3000) -> List[str]:
    """Split text into smaller chunks (approx by words)"""
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
# Summarization with LLM
# =========================
def summarize_chunk(chunk: str, query: str) -> str:
    """Summarize one chunk using LLM"""
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
    """Summarize combined chunk summaries into one final result"""
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
# Main pipeline
# =========================
def main():
    query = input("Enter the topic you want to search for: ")
    num_pages = int(input("Enter number of pages to extract: "))

    all_text = ""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=num_pages)
        for res in results:
            if "href" in res:
                page_text = scrape_page(res["href"])
                all_text += "\n" + page_text

    # Chunk + summarize
    chunks = chunk_text(all_text, max_len=1200)   # adjustable
    print(f"🔹 Created {len(chunks)} chunks")

    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"✍️ Summarizing chunk {i}/{len(chunks)}...")
        chunk_summaries.append(summarize_chunk(chunk, query))

    # Combine + final summarize
    combined = "\n".join(chunk_summaries)
    final_summary = summarize_final(combined, query)

    # Save
    os.makedirs("output", exist_ok=True)
    with open("output/final_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)

    print("\n✅ Extraction and summarization completed.")
    print("Check 'output/final_summary.txt'.")

if __name__ == "__main__":
    main()