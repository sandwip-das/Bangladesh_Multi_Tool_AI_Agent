"""LangChain agent powered by OpenRouter."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

from src.db_tools import get_database_tools
from src.web_search_tool import make_web_search_tool

SYSTEM_PROMPT = """You are a careful Bangladesh information assistant with access to
three local datasets and a web search tool.

Tool policy:
- Questions asking for records, names, counts, locations, or statistics about
  institutions must use institutions_db_tool.
- Questions asking for records, names, counts, or locations about hospitals must
  use hospitals_db_tool.
- Questions asking for records, names, counts, or locations about restaurants
  must use restaurants_db_tool.
- Use the schema tools first when you need to learn the real columns or when a
  SQL query fails. Only use columns that the schema actually contains.
- For general knowledge, policies, definitions, or current information, use
  web_search_tool. Prefer official/primary sources and mention source URLs.
- Do not invent facts or columns. If a dataset lacks the requested field or no
  matching records are found, clearly say so. Do not silently substitute web
  results for database-specific statistics.
- Explain that a count is a count of records in the supplied dataset, not
  necessarily an official or exhaustive count of all real-world entities.
- For SQL, use one read-only SELECT statement, the exact table name, and actual
  column names. Keep queries simple and use LIKE for approximate text matching.
- Give concise, helpful answers. State uncertainty and distinguish dataset facts
  from web-sourced facts.
"""


def build_agent():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Copy .env.example to .env and set it.")

    model_name = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "2000"))
    llm = ChatOpenRouter(
        model=model_name,
        temperature=0,
        max_tokens=max_tokens,
        max_retries=2,
        app_title=os.getenv("OPENROUTER_APP_TITLE", "Bangladesh Multi-Tool AI Agent"),
        app_url=os.getenv("OPENROUTER_APP_URL", "http://localhost"),
    )
    tools = get_database_tools()
    tools.append(make_web_search_tool())
    # Current LangChain 1.x agent API; it returns a compiled graph which
    # supports invoke({"messages": [...]}) and manages tool-calling loops.
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def ask_agent(question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    agent = build_agent()
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = response.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and getattr(message, "content", None):
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
    return str(response)
