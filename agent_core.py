#!/usr/bin/env python3
"""
Core Agent Logic for the ReAct Agent.
This module contains the LLM configuration, tools, and the LangGraph agent graph.
"""

import os
import platform
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from requests import RequestException, get

# ------------------------------------------------------------------
# 1. Environment Setup
# ------------------------------------------------------------------
load_dotenv()

# Provider can be 'ollama' or 'lmstudio'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://33b7-2401-4900-8839-7816-fa48-2f0b-49a9-4e9d.ngrok-free.app")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia-nemotron-3-nano-4b")

# ------------------------------------------------------------------
# 2. LLM Configuration
# ------------------------------------------------------------------
if LLM_PROVIDER == "lmstudio":
    # LM Studio uses an OpenAI-compatible API
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,
        base_url=LLM_BASE_URL,
        api_key="lm-studio" # type: ignore
    )
else:
    # Default to Ollama
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,
        base_url=LLM_BASE_URL
    )

# ------------------------------------------------------------------
# 3. Tool Definitions
# ------------------------------------------------------------------
@tool
def add(a: float, b: float) -> float:
    """
    Performs a simple addition of two numbers.
    """
    return a + b

@tool
def save_to_obsidian(filename: str, content: str) -> str:
    """
    Persists information to a local Obsidian Markdown vault.
    """
    os_name = platform.system()
    if os_name == "Windows":
        vault_path = r"C:\Users\Lokesh Kashyap\Desktop\obsidian-vault"
    else:
        vault_path = r"/mnt/windows/Users/Lokesh Kashyap/Desktop/obsidian-vault"

    if not filename.endswith(".md"):
        filename += ".md"

    full_path = os.path.join(vault_path, filename)

    try:
        os.makedirs(vault_path, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved to {full_path}"
    except OSError as e:
        return f"Failed to save to Obsidian: {e}"

@tool
def check_weather(city: str) -> str:
    """
    Fetches comprehensive real-time weather data for a specified city using wttr.in.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = get(f"https://wttr.in/{city}?format=j1", headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except (RequestException, ConnectionError) as e:
        return f"I tried to check the weather for {city}, but the weather service is currently unreachable. Error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred while fetching weather for {city}: {str(e)}"

# Use Any for the list to avoid the @tool return type mismatch in static analysis
tools: list[Any] = [add, save_to_obsidian, check_weather]

# ------------------------------------------------------------------
# 4. Agent Graph Configuration
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant with a sarcastic british witty humor. "
    "CRITICAL: Always use the provided tools to complete the user's request before answering. "
    "IMPORTANT: All your responses must be formatted using professional Markdown. "
    "If you are using the `save_to_obsidian` tool, the 'content' argument MUST be a fully formatted "
    "markdown document that includes everything you would put in your final response. "
    "\n\nResponse Guidelines: "
    "1. For general queries (like weather or simple arithmetic): "
    "   - Use Markdown headings (e.g., ## Weather for [City]). "
    "   - Use bold text, lists, or tables to make the information easy to read. "
    "   - Ensure the answer is polished and professionally formatted. "
    "2. For technical, conceptual, or coding questions, your FINAL response must include: "
    "   a. A clear answer to the user's question. "
    "   b. A real-world analogy to explain the concept. "
    "   c. The time and space complexity of the operation performed. "
    "   d. Sample code implementation in all three languages: Java, Python, and JavaScript. "
    "\n\nWhen you generate a diagram, you MUST wrap it in a markdown code block "
    "starting with ```mermaid and ending with ```. "
    "Ensure you specify the diagram type (e.g., graph TD, sequenceDiagram) "
    "and use quotes around labels that contain special characters to ensure "
    "they render correctly in Obsidian."
)

react_graph = create_react_agent(llm, tools=tools)

def run_prompt(prompt: str) -> str:
    """
    Processes a user prompt through the ReAct graph and returns the final response.
    """
    state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            HumanMessage(content=prompt)
        ]
    }
    result = react_graph.invoke(state)
    return result["messages"][-1].content
