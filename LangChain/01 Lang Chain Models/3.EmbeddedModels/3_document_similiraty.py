# ===========================
# Import Required Libraries
# ===========================

# .env file se API key load karne ke liye
from dotenv import load_dotenv

# Gemini Embedding Model use karne ke liye
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Cosine Similarity calculate karne ke liye
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# Load Environment Variables (.env file)
# ============================================

# .env file me stored GOOGLE_API_KEY ko load karega
load_dotenv()


# ============================================
# Create Gemini Embedding Model
# ============================================

# Gemini ka embedding model initialize kar rahe hain
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================
# Documents
# ============================================

# Ye documents hamare database ki tarah kaam karenge.
# Inhi documents me se query ke liye best match dhoondhenge.
documents = [

    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",

    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",

    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",

    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",

    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."

]


# ============================================
# User Query
# ============================================

# User kya search kar raha hai
query = "tell me about virat kohli"


# ============================================
# Convert Documents into Embeddings
# ============================================

# Har document ko numerical vector me convert karega
doc_embeddings = embedding.embed_documents(documents)


# ============================================
# Convert Query into Embedding
# ============================================

# Query ko bhi numerical vector me convert karega
query_embedding = embedding.embed_query(query)


# ============================================
# Calculate Cosine Similarity
# ============================================

# Query aur sabhi documents ke beech similarity calculate karega

scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]


# ============================================
# Print Similarity Scores
# ============================================

print("Similarity Scores")
print(scores)


# ============================================
# Sort Similarity Scores
# ============================================

# enumerate() har score ke saath uska index bhi dega
#
# Example:
# [(0,0.91),(1,0.45),(2,0.60)...]
#
# reverse=True matlab highest score pehle aayega

sorted_scores = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1],
    reverse=True
)


print("\nSorted Scores")
print(sorted_scores)


# ============================================
# Best Matching Document
# ============================================

# Sabse pehla element highest similarity wala hoga
best_index, best_score = sorted_scores[0]


# ============================================
# Print Result
# ============================================

print("\n==============================")
print("Query")
print("==============================")
print(query)


print("\n==============================")
print("Most Relevant Document")
print("==============================")
print(documents[best_index])


print("\n==============================")
print("Similarity Score")
print("==============================")
print(best_score)