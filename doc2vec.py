import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Directory containing your pre-tokenized .txt files
directory = '/content/corpora'  # <-- Replace this with your actual path

# Load pre-tokenized data
data = []
file_names = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tokens = content.strip().split()  # assuming tokens are space-separated
            data.append(tokens)
            file_names.append(filename)

# Tag documents for Doc2Vec
tagged_data = [TaggedDocument(words=words, tags=[str(idx)]) for idx, words in enumerate(data)]

# Train Doc2Vec model
model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=1000)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Get vectors
doc_vectors = np.array([model.dv[str(i)] for i in range(len(file_names))])

# Compute cosine similarity
similarity_matrix = cosine_similarity(doc_vectors)

# Print pairwise similarity scores (excluding self-comparisons)
for i in range(len(file_names)):
    for j in range(len(file_names)):
        if i != j:
            print(f"{file_names[i]} <-> {file_names[j]}: {similarity_matrix[i][j]:.4f}")
