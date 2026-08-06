from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of India"

# Store the embedding
vector = embedding.embed_query(text)

print(vector)
print("Embedding Length:", len(vector))