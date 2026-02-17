import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_tavily import TavilySearch
from langchain_xai import ChatXAI

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

# 2. Initialize Grok
llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0)
# using what we imported from our schemas.py and imported from pydantic


output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

system_prompt = react_prompt_with_format_instructions.format(
    tools="TavilySearch", tool_names="TavilySearch", input="", agent_scratchpad=""
)

# 3. Define the Tool
# Using TavilySearch directly
search_tool = [TavilySearch(max_results=3)]

# 4. Build the Agent
# In LangChain 1.0+, 'prompt' is the standard argument for the system message
# system_message = "You are a concise assistant. Use search for current events."
agent_executor = create_agent(model=llm, tools=search_tool, system_prompt=system_prompt)


def ask_agent(question: str):
    """Invoke the agent and extract the final message from the state"""
    # Modern agents return a dict with an 'output' key
    raw_response = agent_executor.invoke({"messages": [("user", question)]})
    final_text = raw_response["messages"][-1].content.strip()
    extract_output = RunnableLambda(lambda text: output_parser.parse(text))
    structured = extract_output.invoke(final_text)
    return structured.answer


# response = agent_executor.invoke({"messages": [("user", question)]})
# return response["messages"][-1].content.strip()


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
