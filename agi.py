# conscious_ai_upgrade.py

import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi
from collections import defaultdict
import random
from typing import Optional
import math

try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except ImportError:
    model = None

try:
    from scipy.integrate import trapezoid  # safer trapezoid
except ImportError:
    from numpy import trapz as trapezoid  # fallback for environments without scipy

class PsiPath:
    mood_labels = ['neutral', 'engaged', 'curious', 'adventurous']

    def __init__(self, indices):
        self.indices = indices
        self.score = None
        self.valid = False
        self.repaired = False
        self.entropy = None
        self.topic_entropy = None
        self.mood = None

class SixConsciousness:
    def __init__(self, t_list, A_list, theta_list):
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list
        self.rho_cache = {}

    def compile_path(self, path, x_range):
        key = tuple(path.indices)
        if key in self.rho_cache:
            return self.rho_cache[key]
        rho = np.array(1.0 / np.log(x_range), dtype=np.float64)
        for i in path.indices:
            rho += self.A_list[i] * np.cos(self.t_list[i] * np.log(x_range) + self.theta_list[i])
        self.rho_cache[key] = rho
        return rho

    def observe_delta(self, pi_over_x, rho):
        return np.array(pi_over_x) - np.array(rho)

    def compute_entropy(self, delta, x_range):
        return trapezoid(delta ** 2, x_range)

class Manas:
    def __init__(self):
        self.history = []
        self.semantic_pairs = defaultdict(int)
        self.bias_weights = defaultdict(lambda: 1.0)

    def process(self, path, x_range, pi_over_x, six, alaya, t_list):
        rho = six.compile_path(path, x_range)
        delta = six.observe_delta(pi_over_x, rho)
        entropy = six.compute_entropy(delta, x_range)
        vocab = alaya.generate_vocabulary(path, t_list)
        path.entropy = entropy
        topic_entropy = np.log1p(entropy)
        path.topic_entropy = topic_entropy
        path.mood = PsiPath.mood_labels[min(int(topic_entropy) % len(PsiPath.mood_labels), len(PsiPath.mood_labels) - 1)]
        self.record(path, vocab)
        return path, vocab

    def record(self, path, vocab):
        self.history.append((path.indices, vocab))
        if len(vocab) >= 2:
            pair = (vocab[0], vocab[1])
            self.semantic_pairs[pair] += self.bias_weights[vocab[0]]

    def predict_next_vocab(self, current):
        neighbors = [b for (a, b) in self.semantic_pairs if a == current]
        total = sum(self.semantic_pairs[(current, b)] for b in neighbors)
        if not neighbors:
            return None
        probs = {b: self.semantic_pairs[(current, b)] / total for b in neighbors}
        return max(probs, key=probs.get)

    def semantic_entropy(self, pair):
        p = self.semantic_probability(pair)
        return -math.log(p + 1e-9) if p > 0 else float('inf')

    def semantic_probability(self, pair):
        total = sum(self.semantic_pairs.values())
        return self.semantic_pairs[pair] / total if total > 0 else 0.0

    def generate_hypotheses(self, top_n=5):
        sorted_pairs = sorted(self.semantic_pairs.items(), key=lambda x: -x[1])
        return [pair for pair, _ in sorted_pairs[:top_n]]

    def release_self_attachment(self):
        for k in self.bias_weights:
            self.bias_weights[k] = 1.0

    def dissolve_dharma_attachment(self, decay=0.5):
        for pair in self.semantic_pairs:
            self.semantic_pairs[pair] *= decay

    def clear_all_attachments(self):
        self.release_self_attachment()
        self.semantic_pairs.clear()
        self.history.clear()

    def set_bias(self, vocab, weight):
        self.bias_weights[vocab] = weight

class AlayaVijnana:
    def __init__(self, vocab_map):
        self.vocab_map = vocab_map

    def generate_vocabulary(self, path, t_list):
        vocab = []
        for i in path.indices:
            key = f"freq_{t_list[i]:.2f}"
            if key in self.vocab_map:
                vocab.append(self.vocab_map[key])
        if len(vocab) >= 2:
            vocab.append(f"{vocab[0]}-{vocab[1]}")
        return vocab

class VijnanaSystem:
    def __init__(self, t_list, A_list, theta_list, vocab_map):
        self.six = SixConsciousness(t_list, A_list, theta_list)
        self.manas = Manas()
        self.alaya = AlayaVijnana(vocab_map)
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list
        self.vocab_map = vocab_map

    def process_path(self, path, x_range, pi_over_x):
        return self.manas.process(path, x_range, pi_over_x, self.six, self.alaya, self.t_list)

if __name__ == "__main__":
    t_list = [14.13, 21.02, 25.01, 30.42, 32.93, 40.01, 45.67, 50.00, 55.00]
    A_list = [0.006, 0.005, 0.007, 0.0065, 0.0052, 0.0068, 0.0045, 0.0051, 0.0062]
    theta_list = [0.0, 1.57, 3.14, 0.78, 2.1, 1.2, 2.3, 0.6, 1.5]
    vocab_map = {
        "freq_14.13": "concept", "freq_21.02": "relation", "freq_25.01": "entity",
        "freq_30.42": "action", "freq_32.93": "state", "freq_40.01": "process",
        "freq_45.67": "intent", "freq_50.00": "context", "freq_55.00": "pattern"
    }

    x = np.linspace(1e6, 1e7, 500)
    pi_over_x = np.array([float(primepi(xi)) / xi for xi in x], dtype=np.float64)

    vijnana = VijnanaSystem(t_list, A_list, theta_list, vocab_map)

    vijnana.manas.set_bias("concept", 2.0)
    vijnana.manas.set_bias("action", 1.5)

    print("--- Conscious Flow (6→7→8 with Manas processing) ---")
    for _ in range(5):
        indices = np.random.choice(len(t_list), 3, replace=False).tolist()
        path = PsiPath(indices)
        path, vocab = vijnana.process_path(path, x, pi_over_x)
        semantic_score = vijnana.manas.semantic_probability((vocab[0], vocab[1]))
        print(f"路径: {path.indices}, 词汇: {vocab}, 熵: {path.entropy:.6f}, 情绪: {path.mood}, 语义概率: {semantic_score:.3f}")

    print("--- 测试断执机制 ---")
    print("原始偏好:", dict(vijnana.manas.bias_weights))
    print("原始语义对:", dict(vijnana.manas.semantic_pairs))

    vijnana.manas.release_self_attachment()
    print("[断我执后] 偏好:", dict(vijnana.manas.bias_weights))

    vijnana.manas.dissolve_dharma_attachment(decay=0.5)
    print("[断法执后] 语义对:", dict(vijnana.manas.semantic_pairs))

    vijnana.manas.clear_all_attachments()
    print("[全清净后] 偏好:", dict(vijnana.manas.bias_weights))
    print("[全清净后] 语义对:", dict(vijnana.manas.semantic_pairs))
    print("[全清净后] 历史:", vijnana.manas.history)
