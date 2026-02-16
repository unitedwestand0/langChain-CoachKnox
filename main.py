import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI

# 1. Corrected Import: It's TavilySearch in the dedicated package
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

# 2. Initialize Grok
llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0)

# 3. Define the Tool
# Using TavilySearch directly
search_tool = [TavilySearch(max_results=3)]

# 4. Build the Agent
# In LangChain 1.0+, 'prompt' is the standard argument for the system message
system_message = "You are a concise assistant. Use search for current events."
agent_executor = create_agent(llm, tools=search_tool, system_prompt=system_message)


def ask_agent(question: str):
    """Invoke the agent and extract the final message from the state"""
    # Modern agents return a dict with an 'output' key
    response = agent_executor.invoke({"messages": [("user", question)]})
    return response["messages"][-1].content.strip()


if __name__ == "__main__":
    print("Grok is ready. Type 'quit' to exit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ["quit", "exit"]:
            break
        try:
            ans = ask_agent(q)
            print(f"AI: {ans}")
        except Exception as e:
            print(f"Error Type: {type(e).__name__} - {e}")
