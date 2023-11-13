import streamlit as st

from app import run, get_providers

providers = get_providers()

st.set_page_config(page_title="💬 Chatbot")

with st.sidebar:
    st.title("💬 Chatbot")

    st.subheader("Models and parameters")
    # selected_db = st.sidebar.selectbox(
    #     label="Choose a database", options=["chromadb", "faiss"], key="selected_db"
    # )
    selected_llm_model = st.sidebar.selectbox(
        label="Choose a LLM",
        options=sorted(providers["llms"].keys()),
        key="selected_llm_model",
    )
    selected_embeddings_model = st.sidebar.selectbox(
        label="Choose a embedding model",
        options=sorted(providers["embeddings"].keys()),
        key="selected_embeddings_model",
    )
    selected_chain_type = st.sidebar.selectbox(
        label="Choose a chain type",
        options=["stuff", "refine", "map_reduce", "map_rerank"],
        key="selected_chain_type",
    )

    authenticate = (
        providers["embeddings"][selected_embeddings_model] == "openai"
        or providers["llms"][selected_llm_model] == "openai"
    )
    openapi_key = st.text_input(
        "Enter OpenAI API key:", type="password", disabled=not authenticate
    )
    block_message = False
    if authenticate:
        if not (openapi_key.startswith("sk-") and len(openapi_key) == 51):
            st.warning("Please enter your credentials!", icon="⚠️")
            block_message = True
        else:
            st.success("Proceed to entering your prompt message!", icon="👉")
    else:
        st.success("Proceed to entering your prompt message!", icon="👉")
    # temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    # top_k = st.sidebar.slider('top_k', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_response(prompt_input):
    output = run(
        chain_type=selected_chain_type,
        llm_model=selected_llm_model,
        embedding_model=selected_embeddings_model,
        collection_name=f"{providers['embeddings'][selected_embeddings_model]}-{selected_embeddings_model}",
        query=prompt_input,
        keys={"OPENAI_API_KEY": openapi_key},
    )
    return output["answer"]


if prompt := st.chat_input(disabled=block_message):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
        placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
