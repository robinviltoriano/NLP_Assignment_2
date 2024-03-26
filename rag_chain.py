from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# model = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":1000})

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

def get_rag_chain(query_answer):
    """
    Des: This function returns a RAG chain which can be used to answer questions based on the given context.
    Params:
    * query_answer: a function that given a query, return a string of context
    """
    
    
    
    def get_context(question):
        print("question", question)
        return query_answer
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()
    
    
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]
    
    qa_system_prompt = """you are an AI assistant. I will give you the given context and our conversation history, and you will have to answer only base on given knowledge, using Natural Language. You can not use your own knowledge or public sources and databases. Things to remember:
                            1. The answer has to be from the given context and our conversation history that I give you.
                            2. You can not use your own knowledge or public sources and databases. This will help to avoid hallucination
                            2. If the answer is not in the given context and our conversation history, just say I don't know
                            Here is the given context and our conversation history:
                        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    def format_docs(conversation):
        print("Conversation:", conversation)
        return conversation

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | query_answer
        )
        | format_docs
        | qa_prompt
        | model
    )
    
    return rag_chain
    