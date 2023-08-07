#######                         Retrival Augmented Generation (RAE)             ##########

# RAE: es el proceso de integración de datos externos (external knowledge base) al LLM.
import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
import openai  # Lo usaremos para enlistar los modelos disponibles y verificar rapidamente la autenticidad del API Key


# Declaracion del titulo
st.title(body="Chatbot")

#########            1) PARAMETRIZACION  Y CARGA DE DATOS    #####################

st.sidebar.subheader(body="Ingreso de datos y configuración")

# Parametros
api_key = st.sidebar.text_input(label="OpenAI API Key", type="password")
documento = st.sidebar.file_uploader(label="Carga del archivo PDF que será ingegrado al LLM "
                                               "(Large Language Model) para su consulta")
chunk_size = st.sidebar.number_input(label="Chunk size:", min_value=100,
                                         max_value=1000, value=400)
chunk_overlap = st.sidebar.number_input(label="Chunk Overlap:", min_value=50,
                                            max_value=500, value=200)






##########                2)  RETRIVAL AUGMENTO GENERATION  (RAE)                #############


# 2.1) CONSTRUCCION DEL CHAT BASE
if "messages" not in st.session_state:
    st.session_state.messages = []  # mensajes vacios

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):

###    2.2)   WALL, no permitira la ejecucion de lo siguiente sino cumple con estos 2 requisitos
    if not api_key:
        st.info("Ingrese un Api-Key")
        st.stop()
    if not documento:
        st.info("Ingrese un documento PDF")
        st.stop()


######  2.3) LOADING THE DOCUMENT:
    reader = PdfReader(stream=documento)

    # Obtén el número total de páginas
    number_of_pages = len(reader.pages)
    all_text = ""  # Declaramos un str vacio

    # Lee el texto de cada página
    for i in range(number_of_pages):
        page = reader.pages[i]
        all_text += page.extract_text()  # ponerle un += es como usar un append en un str



######      2.4) TRANFORM (chunk strategy)

    # Set up del objto text_slpiter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,  # numero de caracteres
        chunk_overlap=chunk_size,  # solapamiento
        length_function=len)

    # IMPORTANTE: Con RecursiveCharacterTextSplitter podemos usar length function, esta funcion "len" cunta solo los caraceres. PEro podriamos usar en lugar de esta funcion alguna otra de contabiliszacion de tokens.  Cualquiera  haria una dividion de chunks, exsten otro modulo que es exclusivo para dividir los textos solo  por caracteres

    # Ejecucion del objto text spliter
    # Se tiene que ingresar solo string
    chunks = text_splitter.create_documents(texts=[all_text])  # Este lo ejecutamos solo con el string



##########        2.5)  EMBEDDING
    # En este caso estamos integrando un embedding modelo de OpenAI, set up el embedding model
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)


###########      2.6) VECTOR DATABASE

    # Creamos nuestro Vectordatabse, le integramos los texgtos (chunkgs) y el modelo de embedding que se usara
    db = Chroma.from_documents(documents=chunks, embedding=embeddings_model)

    # Creamos el objeeto Vector Store Retriver
    retriever = db.as_retriever()



#############      2.7)  CREACION E INTEGRACION DE UN TEMPLATE
    prompt_template = """Use the following pieces of context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.You are a chatbot dedicate to solve questions about the company Empresa X.
    
    {context}
    
    Question: {question}
    Answer in spanish:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"])



#########  2.8 ) SET UP DEL LLM
    llm_openai = OpenAI(openai_api_key=api_key,
                        model_name="gpt-3.5-turbo",
                        temperature=0.2)  # Entre mayor la temperatura más random las respuestas



#########     2.9) CREACION DEL CHAIN que integra todo: el vector database (con el retriver), el llm y su prompt
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm_openai, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs)


#######        2.10) CREACION DE LA CONVERSACION DEL CHAT
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = qa.run(prompt)  ##   EJECUCION DEL LLM
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})














