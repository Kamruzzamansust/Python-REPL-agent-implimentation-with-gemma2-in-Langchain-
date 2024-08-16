from langchain_groq import ChatGroq
from langchain import hub
from dotenv import load_dotenv
import os 
from langchain.agents import create_react_agent , AgentExecutor
from langchain_experimental.tools import PythonREPLTool
import pyboxen
from pyboxen import boxen
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-It",
        temperature=0.5
    )


instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """


base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

tools = [PythonREPLTool()]
agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools,
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke(
        input={
            "input": """please Give me an example of List comprhension in python . where if else is used. """
        }
    )

print(boxen(result['output'],color="cyan",margin=1, padding=1))