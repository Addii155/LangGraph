from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Setup Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

# Define the state schema
class AgentState(TypedDict):
    message: List[HumanMessage]

# Define the processing node
def process(state: AgentState) -> AgentState:
    response = model.invoke(state["message"])
    print(f"\n🔹 AI Response: {response.content}")
    return state

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")

# Compile
app = graph.compile()

# Invoke with proper state
user_input=input("\nYOU: ")
while user_input!="exit":
    app.invoke({"message":[HumanMessage(content=user_input)]})  
    user_input=input("\nYOU: ")
