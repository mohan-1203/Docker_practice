from sentence_transformers import SentenceTransformer
import faiss


# Load knowledge document
with open("knowledge/student_rules.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Create chunks
chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print("Number of chunks:", len(chunks))


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create embeddings
embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)


# Make sure embeddings are 2-dimensional
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)

embeddings = embeddings.astype("float32")


print("Embedding shape:", embeddings.shape)


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


def retrieve_context(question, top_k=2):

    # Create embedding for user question
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")


    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )


    results = []

    for i in indices[0]:

        if i < len(chunks):
            results.append(chunks[i])


    return results