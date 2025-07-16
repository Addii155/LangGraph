from typing import TypedDict, List , Union
from langchain_core.messages import HumanMessage , AIMessage
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
    message: List[Union[AIMessage,HumanMessage]]

def process(state:AgentState)->AgentState:
    response=model.invoke(state["message"])
    state["message"].append(AIMessage(content=response.content))
    print(f"\n {response.content}\n")
    # print(state["message"])

graph = StateGraph(AgentState)

graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)
app= graph.compile()
user_input = input("ENTER: ")
conversation_message=[]
while user_input!='exit':
    conversation_message.append(HumanMessage(content=user_input))
    result= app.invoke({"message":conversation_message})
    # print(result)
    conversation_message=result['message']
    user_input=input("ENTER: ")

with open('logfile.txt','w') as file:

    for message in conversation_message:
        if isinstance(message,HumanMessage):
            file.write(f"YOU: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("end of conversation")
print("Conversation saved to logging.txt")




