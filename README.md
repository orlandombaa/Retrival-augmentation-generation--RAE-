# Retrival augmentation generation (RAE)

This repo contains a web app made with Streamlit. it integrates an LLM from OpenAI using Langchain for exploring an external document, this approach is called Retrieval augmentation generation (RAE).

In general terms, the main steps for this  can be divided into the following steps: 
1) Loading the document
2) Transforming the document (in this case a chunk strategy based on lengths of characters)
3) Set up the embedding model
4) Create the vector database
5) Set up the LLM model
6) Creation of the global chain

 You can check the code in order to see in more details the process. In case you want to take a look how it works you can find in the following link: 

 https://wjmdqv5uk2lxvg243ssrjr.streamlit.app/
