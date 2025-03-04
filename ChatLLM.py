from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM

def query_llm_cot(question, db, model_selected_llm):
    """Query the LLM using relevant document context with an advanced chain-of-thought approach."""
    
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model=model_selected_llm)

    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])

    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    sources = [doc.metadata.get("source", "Unknown") for doc in retrieved_docs]

    # **Step 1: Generate Reasoning Plan**
    reasoning_steps_prompt = f"""For the given question, generate a sequence of reasoning steps to follow.

    ### Question:
    {question}

    ---
    ### Step-by-Step Reasoning Plan:
    """
    reasoning_steps = llm.invoke(reasoning_steps_prompt)

    # **Step 2: Execute Reasoning Steps**
    execution_prompt = f"""Execute each reasoning step below using the retrieved context.

    ### Reasoning Steps:
    {reasoning_steps}

    ### Retrieved Context:
    {formatted_context}

    ---
    ### Answer:
    """
    initial_response = llm.invoke(execution_prompt)

    # **Step 3: Self-Verification & Critique**
    critic_system_messsage = SystemMessage(f"""You are to evaluate the response using the following checklist.

    **Checklist:**
    1. Is the response fully supported by the retrieved context?
    2. Did the response answer all subparts of the question?
    3. Does the reasoning follow a logical sequence?
    4. Is any claim made without evidence?
    5. Is the answer clear, concise, and well-structured?
    
    ---
    Provide a revised response if any issues are found.
    """)

    critic_prompt = HumanMessage(f"""
    ### Retrieved Context:
    {formatted_context}

    ### Question:
    {question}

    ### Initial Response:
    {initial_response}
    """)

    critic_message = [critic_system_messsage, critic_prompt]
    final_response = llm.invoke(critic_message)

    print("\nSources Used:")
    for source in sources:
        print(f"- {source}")

    return {"answer": final_response, "sources": sources}
