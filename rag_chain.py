from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores.faiss import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)




model = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1000})

def get_rag_chain(retriever):
    """
    Des: This function returns a RAG chain which can be used to answer questions based on the given context.
    Params:
    * retriever: a function that given a query, return 
    
    Example usecase:
        chat_history = []

        question = "What is the weight of this assignment?"
        ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=question), ai_msg])

        second_question = "How many of them is from report?"
        ai_msg = rag_chain.invoke({"question": second_question, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=second_question), ai_msg])
        
        >>> chat_history
        >>> [HumanMessage(content='What is the weight of this assignment?'),
            '30% of this assignment weigh ng is for the code, and 70% for the report. The code and the report are marked per rubric.',
            HumanMessage(content='How many of them is from report?'),
            '70%']
    """
    
    def get_context(sth):
        print(retriever, sth)
        return retriever
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
                                    which might reference context in the chat history, formulate a standalone question \
                                    which can be understood without the chat history. Do NOT answer the question, \
                                    just reformulate it if needed and otherwise return it as is. But the most important thing, you have to answer using natural language as a human.{chat_history}"""
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
            print(contextualize_q_chain)
            return contextualize_q_chain
        else:
            return input["question"]
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
                        Use the following pieces of retrieved context to answer the question. \
                        If you don't know the answer, just say that you don't know. But the most important thing, you have to answer using natural language as a human.\

                        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | RunnableLambda(get_context)
        )
        | qa_prompt
        | model
    )
    
    return rag_chain
    