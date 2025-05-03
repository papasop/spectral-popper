# conscious_ai_upgrade.py

import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi
from collections import defaultdict
import random
from typing import Optional
try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except ImportError:
    model = None
import math
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

class MemoryChain:
    def __init__(self):
        self.history = []
        self.semantic_pairs = defaultdict(int)

    def generate_hypotheses(self, top_n=5):
        sorted_pairs = sorted(self.semantic_pairs.items(), key=lambda x: -x[1])
        return [pair for pair, _ in sorted_pairs[:top_n]]

    def record(self, path, vocab):
        self.history.append((path.indices, vocab))
        if len(vocab) >= 2:
            pair = (vocab[0], vocab[1])
            self.semantic_pairs[pair] += 1

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

class StructureLangPlanner:
    def generate_path_with_vocab_pair(self, vocab1, vocab2):
        indices = []
        for i, t in enumerate(self.t_list):
            key = f"freq_{t:.2f}"
            if key in self.vocab_map and self.vocab_map[key] in (vocab1, vocab2):
                indices.append(i)
            if len(indices) >= 2:
                break
        if len(indices) < 2:
            return None
        extras = [i for i in range(len(self.t_list)) if i not in indices]
        if extras:
            indices.append(random.choice(extras))
        return PsiPath(indices)

    def __init__(self, t_list, A_list, theta_list, vocab_map=None):
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list
        self.vocab_map = vocab_map or {}
        self.history = []
        self.rho_cache = {}

    def compile(self, path, x_range):
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

    def validate(self, path, score_threshold=0.006):
        A = [self.A_list[i] for i in path.indices]
        path.score = np.mean(A)
        path.valid = path.score > score_threshold
        return path.valid

    def repair(self, path, x_range, pi_over_x, max_attempts=50):
        best_path = path
        best_score = path.score or 0.0
        best_entropy = float("inf")
        temperature = 5.0
        seen = set()

        for attempt in range(max_attempts):
            new_indices = np.random.choice(len(self.t_list), len(path.indices), replace=False).tolist()
            if tuple(sorted(new_indices)) in seen:
                continue
            seen.add(tuple(sorted(new_indices)))
            temp_path = PsiPath(new_indices)

            if not self.validate(temp_path):
                continue

            rho = self.compile(temp_path, x_range)
            delta = self.observe_delta(pi_over_x, rho)
            entropy = trapezoid(delta ** 2, x_range)
            temp_path.entropy = entropy

            diversity_penalty = len(set(temp_path.indices) & set(path.indices)) / len(path.indices)
            adjusted_score = entropy + diversity_penalty * 10

            if adjusted_score < best_entropy or random.random() < np.exp(-(adjusted_score - best_entropy) / temperature):
                best_path = temp_path
                best_entropy = adjusted_score
                best_score = temp_path.score

            temperature *= 0.9

        best_path.repaired = True
        return best_path

    def generate_vocabulary(self, path):
        vocab = []
        for i in path.indices:
            key = f"freq_{self.t_list[i]:.2f}"
            if key in self.vocab_map:
                vocab.append(self.vocab_map[key])
        if len(vocab) >= 2:
            vocab.append(f"{vocab[0]}-{vocab[1]}")
        return vocab

    def semantic_similarity(self, dialogue: str) -> Optional[float]:
        if model:
            ref = f"{dialogue.split()[1]} relates to {dialogue.split()[-2]}"
            return float(util.cos_sim(model.encode(ref, convert_to_tensor=True), model.encode(dialogue, convert_to_tensor=True))[0])
        return None

    def compare_paths(self, path_a, path_b):
        # Compare paths using both entropy and semantic coherence
        def score(path):
            vocab = self.generate_vocabulary(path)
            if len(vocab) >= 2:
                sem_prob = memory.semantic_probability((vocab[0], vocab[1]))
            else:
                sem_prob = 0.0
            return path.entropy + (1 - sem_prob) * 5  # weighted
        return score(path_a), score(path_b)

    def parallel_generate_paths(self, x_range, pi_over_x, N=3, count=5):
        results = []
        seen = set()
        attempts = 0
        max_attempts = count * 3
        while len(results) < count and attempts < max_attempts:
            indices = np.random.choice(len(self.t_list), N, replace=False).tolist()
            key = tuple(sorted(indices))
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            path = PsiPath(indices)
            if not self.validate(path):
                path = self.repair(path, x_range, pi_over_x)
            rho = self.compile(path, x_range)
            delta = self.observe_delta(pi_over_x, rho)
            entropy = trapezoid(delta ** 2, x_range)
            topic_entropy = np.log1p(entropy)
            mood_index = min(int(topic_entropy) % len(PsiPath.mood_labels), len(PsiPath.mood_labels) - 1)
            path.entropy = entropy
            path.topic_entropy = topic_entropy
            path.mood = PsiPath.mood_labels[mood_index]
            results.append(path)
        results.sort(key=lambda p: (p.entropy if p.entropy is not None else float('inf'), -p.score if p.score is not None else 0.0))
        return results

def plot_entropy_trend(paths):
    print("\n--- Hypothesis Evaluation ---")
    best_score = float('inf')
    best_info = None
    for pair in memory.generate_hypotheses():
        path = planner.generate_path_with_vocab_pair(*pair)
        if path:
            if not planner.validate(path):
                path = planner.repair(path, x, pi_over_x)
            rho = planner.compile(path, x)
            delta = planner.observe_delta(pi_over_x, rho)
            entropy = trapezoid(delta ** 2, x)
            sem_entropy = memory.semantic_entropy(pair)
            score = entropy + sem_entropy
            vocab = planner.generate_vocabulary(path)
            print(f"假设: {pair[0]} → {pair[1]} | 路径: {path.indices}, 总熵: {score:.4f} (结构 {entropy:.4f} + 语义 {sem_entropy:.4f}), 词汇: {vocab}")
            if score < best_score:
                best_score = score
                best_info = (pair, path, vocab, score)
    if best_info:
        pair, path, vocab, score = best_info
        dialogue = f"Following {vocab[0].capitalize()}, {vocab[0]} {random.choice(['relates to', 'transforms into', 'initiates', 'supports', 'questions'])} {vocab[1]} with high semantic plausibility."
        print(f"\n✅ 最佳假设路径: {path.indices}, 词汇: {vocab}, 总熵: {score:.4f}\n🧠 Dialogue: {dialogue}")

    entropies = [p.entropy for p in paths]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(entropies)+1), entropies, marker='o')
    plt.title("路径结构熵趋势")
    plt.xlabel("路径序号")
    plt.ylabel("结构熵")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

    memory = MemoryChain()
    planner = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)
    paths = planner.parallel_generate_paths(x, pi_over_x, N=3, count=5)

    print("\n--- Semantic Path Inference ---")
    for i, path in enumerate(paths):
        vocab = planner.generate_vocabulary(path)
        dialogue = f"Following {vocab[0].capitalize()}, {vocab[0]} {random.choice(['relates to', 'transforms into', 'initiates', 'supports', 'questions'])} {vocab[1]} with mood {path.mood}."
        memory.record(path, vocab)
        semantic_score = memory.semantic_probability((vocab[0], vocab[1]))
        print(f"[{i+1}] 路径: {path.indices}, 词汇: {vocab}, 熵: {path.entropy:.6f}, 主题熵: {path.topic_entropy:.3f}, 情绪: {path.mood}, 语义概率: {semantic_score:.3f} 🗣️ Dialogue: {dialogue}")

    plot_entropy_trend(paths)

    print("\n--- Prediction-Guided Path Generation ---")
    for node in set(a for (a, _) in memory.semantic_pairs):
        predicted = memory.predict_next_vocab(node)
        if predicted:
            guided_path = planner.generate_path_with_vocab_pair(node, predicted)
            if guided_path:
                guided_vocab = planner.generate_vocabulary(guided_path)
                print(f"✓ 推荐路径: {guided_path.indices} -> 词汇: {guided_vocab}")
