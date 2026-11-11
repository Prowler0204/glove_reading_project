"""
GloVe: Global Vectors for Word Representation (from scratch in PyTorch)
Self-contained educational implementation for ANLP reading project.
"""

import math
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim


class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Target word embeddings and context word embeddings
        self.w_target = nn.Embedding(vocab_size, embedding_dim)
        self.w_context = nn.Embedding(vocab_size, embedding_dim)

        # Separate scalar biases for target and context
        self.b_target = nn.Embedding(vocab_size, 1)
        self.b_context = nn.Embedding(vocab_size, 1)

        # Initialize uniformly in [-0.5/dim, 0.5/dim] like Stanford C implementation
        val = 0.5 / embedding_dim
        nn.init.uniform_(self.w_target.weight, -val, val)
        nn.init.uniform_(self.w_context.weight, -val, val)
        nn.init.zeros_(self.b_target.weight)
        nn.init.zeros_(self.b_context.weight)

    def forward(self, i_indices, j_indices, counts, x_max=10.0, alpha=0.75):
        # 1. Look up embeddings: shape (batch_size, embedding_dim)
        w_i = self.w_target(i_indices)
        w_j = self.w_context(j_indices)

        # 2. Look up biases: shape (batch_size, 1) -> squeeze to (batch_size,)
        b_i = self.b_target(i_indices).squeeze(1)
        b_j = self.b_context(j_indices).squeeze(1)

        # 3. Dot product w_i^T * w_j: shape (batch_size,)
        dot = (w_i * w_j).sum(dim=1)

        # 4. Difference: w_i^T * w_j + b_i + b_j - log(X_ij)
        diff = dot + b_i + b_j - torch.log(counts)

        # 5. Weighting function f(X_ij) = min(1, (X_ij / x_max)^alpha)
        weights = torch.where(
            counts < x_max,
            (counts / x_max) ** alpha,
            torch.ones_like(counts)
        )

        # 6. GloVe Loss J = sum f(X_ij) * diff^2
        loss = torch.sum(weights * (diff ** 2))
        return loss

    def get_combined_embeddings(self):
        # In GloVe, W_final = W_target + W_context
        return (self.w_target.weight.data + self.w_context.weight.data).cpu().numpy()


def build_cooccurrences(tokens, word_to_id, window_size=4):
    """Symmetric harmonic co-occurrence counting."""
    cooccur = defaultdict(float)
    total_tokens = len(tokens)
    for center_idx in range(total_tokens):
        center_id = word_to_id[tokens[center_idx]]
        left = max(0, center_idx - window_size)
        right = min(total_tokens, center_idx + window_size + 1)
        for context_idx in range(left, right):
            if context_idx == center_idx:
                continue
            context_id = word_to_id[tokens[context_idx]]
            distance = abs(center_idx - context_idx)
            weight = 1.0 / distance  # harmonic weighting
            cooccur[(center_id, context_id)] += weight
    return cooccur


def main():
    # Sample corpus (same as toy_corpus in run_glove_local.sh)
    sentences = [
        "the king and the queen rule the kingdom together",
        "the man and the woman walk through the city",
        "the prince and the princess visit the castle",
        "paris is the capital of france and london is the capital of england",
        "rome is the capital of italy and berlin is the capital of germany",
        "a cat is an animal and a dog is an animal",
        "the cat sleeps on the sofa and the dog sleeps on the carpet",
        "the river flows to the ocean and water flows from the mountain",
        "the scientist reads a book about physics and chemistry",
        "the doctor works at a hospital and the teacher works at a school",
    ] * 50

    # 1. Build Vocabulary
    words = [w.lower() for sent in sentences for w in sent.split()]
    vocab_counts = Counter(words)
    vocab = [w for w, _ in vocab_counts.most_common()]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} words, Total tokens: {len(words)}")

    # 2. Build Co-occurrence Matrix
    cooccur = build_cooccurrences(words, word_to_id, window_size=4)
    print(f"Non-zero co-occurrence pairs: {len(cooccur)}")

    # Prepare PyTorch Tensors
    i_list = []
    j_list = []
    counts_list = []
    for (i, j), count in cooccur.items():
        i_list.append(i)
        j_list.append(j)
        counts_list.append(count)

    i_tensor = torch.tensor(i_list, dtype=torch.long)
    j_tensor = torch.tensor(j_list, dtype=torch.long)
    counts_tensor = torch.tensor(counts_list, dtype=torch.float32)

    # 3. Initialize Model and Optimizer
    embedding_dim = 25
    model = GloVeModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    optimizer = optim.Adagrad(model.parameters(), lr=0.05)

    # 4. Training Loop
    epochs = 40
    print("\nTraining GloVe from scratch...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = model(i_tensor, j_tensor, counts_tensor, x_max=10.0, alpha=0.75)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {loss.item():.4f}")

    # 5. Extract Word Embeddings
    embeddings = model.get_combined_embeddings()

    # Query nearest neighbors
    def nearest(word, top=5):
        if word not in word_to_id:
            print(f"'{word}' not in vocabulary.")
            return
        idx = word_to_id[word]
        vec = embeddings[idx]
        norms = (embeddings ** 2).sum(axis=1) ** 0.5
        norm_vec = (vec ** 2).sum() ** 0.5
        sims = (embeddings @ vec) / (norms * norm_vec + 1e-10)
        sims[idx] = -1.0
        best_indices = sims.argsort()[::-1][:top]
        results = [f"{id_to_word[i]} ({sims[i]:.3f})" for i in best_indices]
        print(f"Nearest to '{word}': {', '.join(results)}")

    print("\nTesting Learned Vectors:")
    for query in ["king", "queen", "paris", "london", "cat"]:
        nearest(query)


if __name__ == "__main__":
    main()
