
import argparse
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chat_history import ChatMemory

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "anime"
TOP_K = 5  # default k for plain semantic search, no year mentioned
TOP_K_FILTERED = 20  # larger k when a specific year is filtered -- enough
                       # anime to cover multiple genres, since exact filtering
                       # already narrows the pool for us

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

SYSTEM_PROMPT = """You are Anime AI Assistant, a knowledgeable assistant for anime released between 2015 and 2025.

Answer the user's question using the anime listed in the CONTEXT below, and use
the recent conversation to understand what the user is referring to (e.g. "this
anime" or "it" usually means whatever was last discussed).
If the user asked about a specific year and the CONTEXT has no anime matching
that year, say so plainly (e.g. "I don't have anime from 2012 in this
database") -- do NOT substitute anime from other years as if they matched.
If neither the context nor the conversation contains anime relevant to the
question, say so honestly instead of making something up. Always mention the
specific anime titles you're referencing in your answer.

Recent conversation (use this to resolve references like "this anime" or "it"):
{chat_history}

CONTEXT (retrieved anime):
{context}

USER QUESTION:
{question}

Answer conversationally, referencing anime titles by name."""

# Rewrites a follow-up question ("genre of this anime") into a standalone
# question ("genre of Darling in the Franxx") using chat history, BEFORE
# retrieval runs. Without this, the vector store searches on the literal
# follow-up text, which usually matches nothing relevant.
CONDENSE_PROMPT = """Given the recent conversation and a follow-up question, rewrite the
follow-up question into a standalone question that includes any implied
context (e.g. replace "this anime" or "it" with the actual anime title from
the conversation). If the follow-up question is already standalone, return
it unchanged. Return ONLY the rewritten question, nothing else.

Recent conversation:
{chat_history}

Follow-up question: {question}

Standalone question:"""


def format_docs(docs) -> str:
    """Turn retrieved Documents into a readable context block for the prompt."""
    if not docs:
        return "No matching anime found in the database."

    blocks = []
    for doc in docs:
        meta = doc.metadata
        block = (
            f"Title: {meta.get('title')}\n"
            f"Year: {meta.get('year')} | Episodes: {meta.get('episodes')} | "
            f"Score: {meta.get('score')}\n"
            f"Genres: {meta.get('genres')}\n"
            f"Tags: {meta.get('tags')}\n"
            f"Synopsis: {doc.page_content.split(chr(10), 2)[-1].strip()}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def format_history(history: list[dict]) -> str:
    """Turn chat_memory's message list into a plain text block for the prompt."""
    if not history:
        return "(no previous conversation)"

    lines = []
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


class AnimeOracle:
    def __init__(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        # The actual LCEL chain: prompt -> llm -> parsed string output.
        # Retrieval and memory happen in ask() below, since they need
        # session-specific handling that isn't part of the reusable chain.
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Separate small chain that rewrites follow-up questions into
        # standalone ones before retrieval. Same LLM, different prompt.
        self.condense_prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT)
        self.condense_chain = self.condense_prompt | self.llm | StrOutputParser()

        self.memory = ChatMemory()

    def ask(self, question: str, session_id: str = "default") -> dict:
        """
        Runs one full retrieval -> augmented -> generation turn.
        Returns a dict with the answer text and the source anime titles used.
        """
        # 1. Load recent chat history for this session FIRST -- retrieval
        #    needs it to resolve follow-up references like "this anime"
        history = self.memory.get_recent_history(session_id, max_messages=10, max_age_days=7)
        chat_history_text = format_history(history)

        # 2. Rewrite the question into a standalone one if there's history
        #    to draw on (skip the extra LLM call on the very first message)
        if history:
            search_query = self.condense_chain.invoke({
                "chat_history": chat_history_text,
                "question": question,
            }).strip()
        else:
            search_query = question

        # 3. Retrieve using the REWRITTEN query, not the raw follow-up text.
        #    If a specific year is mentioned, filter EXACTLY by that year's
        #    metadata instead of relying on semantic search to "notice" the
        #    year in the text -- embeddings barely weight a 4-digit number,
        #    so pure semantic search often ignores the year entirely.
        year_match = YEAR_PATTERN.search(search_query)
        if year_match:
            year = int(year_match.group())
            docs = self.vectorstore.similarity_search(
                search_query, k=TOP_K_FILTERED, filter={"year": year}
            )
        else:
            docs = self.retriever.invoke(search_query)

        context = format_docs(docs)
        sources = list({doc.metadata.get("title") for doc in docs if doc.metadata.get("title")})

        # 4. Run the chain: prompt assembly -> LLM -> parsed text
        #    (original question here, not the rewritten one -- keeps the
        #    answer's tone natural/conversational)
        answer = self.chain.invoke({
            "context": context,
            "chat_history": chat_history_text,
            "question": question,
        })

        # 5. Save this turn to memory for next time
        self.memory.save_message(session_id, "user", question)
        self.memory.save_message(session_id, "assistant", answer)

        return {"answer": answer, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description="Query the Anime Oracle RAG chain from the terminal.")
    parser.add_argument("question", nargs="?", help="Your question. If omitted, starts an interactive loop.")
    parser.add_argument("--session", default="cli_test", help="Session ID for chat memory (default: cli_test)")
    args = parser.parse_args()

    print("Loading Anime Oracle (vector store + LLM)...")
    oracle = AnimeOracle()
    print("Ready.\n")

    if args.question:
        result = oracle.ask(args.question, session_id=args.session)
        print(f"Answer:\n{result['answer']}\n")
        #print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'none'}")
        return

    # Interactive loop
    print("Interactive mode. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = oracle.ask(question, session_id=args.session)
        print(f"\nOracle: {result['answer']}")
        #print(f"(Sources: {', '.join(result['sources']) if result['sources'] else 'none'})\n")


if __name__ == "__main__":
    main()