from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the model
def load_llm():
    try:
        llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# QA Model Function
def qa_bot():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        return qa
    except Exception as e:
        print(f"Error creating QA bot: {e}")
        return None

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    if not chain:
        msg = cl.Message(content="Failed to initialize the bot. Please try again later.")
        await msg.send()
        return

    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Ecom Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Session expired. Please restart the bot.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: {str(sources)}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()