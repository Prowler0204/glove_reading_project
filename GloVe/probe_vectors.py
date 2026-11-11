#!/usr/bin/env python3
"""Readable nearest-neighbor + analogy demo for Stanford GloVe vectors.txt.
Usage: python3 probe_vectors.py vectors.txt king queen paris london
Requires: pip install numpy
Does not modify Stanford's training code.
"""
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('file', nargs='?', default='vectors.txt')
    p.add_argument('words', nargs='*', default=['king', 'queen', 'man', 'woman'])
    p.add_argument('--top', type=int, default=8)
    a = p.parse_args()
    words, vecs = [], []
    with open(a.file, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) <= 2 or parts[0] == '<unk>':
                continue
            words.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])
    if not words:
        raise SystemExit('No vectors found. Did training finish?')
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-10)
    lookup = {w:i for i,w in enumerate(words)}
    print(f'Loaded {len(words):,} vocabulary words; vector dimension {mat.shape[1]}.')
    for word in a.words:
        if word not in lookup:
            print(f'\n{word}: not in vocabulary (this is normal on a tiny corpus)')
            continue
        score = mat @ mat[lookup[word]]
        score[lookup[word]] = -np.inf
        indices = np.argsort(-score)[:a.top]
        print(f'\nNearest to {word}: ' + ', '.join(f'{words[i]} ({score[i]:.3f})' for i in indices))
    if all(x in lookup for x in ['king','man','woman']):
        target = mat[lookup['king']] - mat[lookup['man']] + mat[lookup['woman']]
        target = target / max(np.linalg.norm(target), 1e-10)
        score = mat @ target
        for x in ['king','man','woman']:
            score[lookup[x]] = -np.inf
        idx = np.argsort(-score)[:a.top]
        print('\nAnalogy king - man + woman: ' + ', '.join(f'{words[i]} ({score[i]:.3f})' for i in idx))
        print('Note: the answer depends on corpus size/training; it is not guaranteed to be queen.')

if __name__ == '__main__':
    main()
