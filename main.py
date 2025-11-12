from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
#"anthropic:claude-sonnet-4-5-20250929" also possible as modern alternative

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state:State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([{
                "role": "system",
        "content": """Classify the user message as either:
        - 'emotional' if it asks for emotional support, therapy, deals with feelings
        - 'logical': if it asks for facts, information, logical analysis, or practical advice"""
      },
      {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}

def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages =[
        {"role": "system",
          "content": """You are a compassionate therapise. Focus on emotional aspects.
          Show empathy, validate their feelings and help them process their emotions.
          Ask thoughtful questions to help them explore their feelings more.
          Avoid giving logical solutions unless explicitly asked."""
         },
         {
             "role": "user",
             "content": last_message.content
         }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages =[
        {"role": "system",
          "content": """You are a logical agent. Focus only on facts and information.
          Provide clear answer based on logic and evidence."""
         },
         {
             "role": "user",
             "content": last_message.content
         }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}

graph_builder = StateGraph(State)

#Nodes
graph_builder.add_node("classifier", classify_message)

graph_builder.add_node("router", router)

graph_builder.add_node("therapist", therapist_agent)

graph_builder.add_node("logical", logical_agent)
#Edges
graph_builder.add_edge(START,"classifier")

graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges("router",
        lambda state: state.get("next"),
        {"therapist": "therapist", "logical": "logical"}
    )

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()

state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()