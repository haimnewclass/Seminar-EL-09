import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'eee'




# Step 1: Load the text file
loader = TextLoader(r'C:\c\story.txt')
documents = loader.load()

# Step 2: Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Step 3: Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings()

# Step 4: Generate embeddings for the text chunks
text_embeddings = [embeddings.embed_query(doc.page_content) for doc in texts]

# Step 5: Convert embeddings to a numpy array
embeddings_array = np.array(text_embeddings)

# Step 6: Create a FAISS index
index = faiss.IndexFlatL2(embeddings_array.shape[1])
index.add(embeddings_array)

# Step 7: Function to search the FAISS index
def search(query, k=5):
    query_embedding = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    return [texts[i].page_content for i in indices[0]]

# Step 8: Initialize the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Or use "gpt-4"

# Step 9: Perform a search
query = "Your search query here"
results = search(query)

# Step 10: Use the chat model to generate a response
# The input needs to be structured as messages for a chat model
messages = [{"role": "system", "content": "You are a helpful assistant."}]
for result in results:
    messages.append({"role": "user", "content": result})

# Use the invoke method (updated API)
response = llm.invoke(messages)

# Print the response content
print(response.content)