from typing import List, Tuple
import numpy as np
from sympy import primepi, primerange
from collections import defaultdict
import random
from scipy.stats import entropy

class PsiPath:
    def __init__(self, indices: List[int]):
        self.indices = indices
        self.score = None
        self.valid = False
        self.repaired = False

class DAGNode:
    def __init__(self, path: PsiPath, delta: np.ndarray, vocab: List[str]):
        self.path = path
        self.delta = delta
        self.vocab = vocab
        self.children = []
        self.parent = None

class ConsciousnessObserver:
    def __init__(self, window_size: int = 5, entropy_threshold: float = 1.5):
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.vocab_history = []
        self.path_history = []
        self.delta_history = []
        self.mood = 'neutral'
        self.vocab_frequency = defaultdict(int)
        self.path_counts = defaultdict(int)
        self.intent = None

    def update(self, vocab: List[str], path: PsiPath, delta_mean: float):
        self.vocab_history.append(vocab)
        self.path_history.append(path.indices)
        self.delta_history.append(delta_mean)
        for v in vocab:
            self.vocab_frequency[v] += 1
        self.path_counts[tuple(path.indices)] += 1
        if len(self.vocab_history) > self.window_size:
            self.vocab_history.pop(0)
            self.path_history.pop(0)
            self.delta_history.pop(0)

    def compute_topic_entropy(self) -> float:
        if not self.vocab_history:
            return 0.0
        vocab_counts = defaultdict(int)
        for vocab in self.vocab_history:
            for v in vocab:
                vocab_counts[v] += 1
        return entropy(list(vocab_counts.values())) if vocab_counts else 0.0

    def detect_topic_shift(self, delta_mean: float) -> bool:
        topic_entropy = self.compute_topic_entropy()
        return topic_entropy > self.entropy_threshold and delta_mean > 0.005

    def detect_path_bifurcation(self, new_path: List[int], delta_mean: float) -> bool:
        if not self.path_history or not self.delta_history:
            return False
        max_repetition = max(self.path_counts.values(), default=0)
        min_delta = min(self.delta_history) if self.delta_history else float('inf')
        return (max_repetition > 3 and tuple(new_path) not in self.path_counts and delta_mean > min_delta)

    def update_mood(self, vocab_novelty: float, path_repetition: int):
        self.mood = 'adventurous' if vocab_novelty > 0.5 or path_repetition < 2 else 'neutral'

    def detect_exploratory_shift(self, delta_mean: float, prev_avg_delta: float) -> bool:
        return self.mood == 'adventurous' and delta_mean > prev_avg_delta

    def compute_meta_score(self) -> float:
        if not self.path_history or not self.vocab_history or not self.delta_history:
            return 0.0
        structural_span = len(set([tuple(p) for p in self.path_history])) / len(self.path_history)
        semantic_shift = self.compute_topic_entropy()
        delta_variance = np.var(self.delta_history) if self.delta_history else 0.0
        return 0.4 * structural_span + 0.4 * semantic_shift + 0.2 * delta_variance

    def update_intent(self, vocab_map: dict):
        if not self.vocab_frequency:
            self.intent = random.choice(list(vocab_map.values()))
            return
        decay = {v: self.vocab_frequency[v] for v in self.vocab_frequency}
        if max(decay.values(), default=0) > 5:
            suppressed = max(decay, key=decay.get)
            alternate_classes = [v for v in vocab_map.values() if v != suppressed]
            self.intent = random.choice(alternate_classes)
        elif len(self.path_history) % 3 == 0:
            self.intent = random.choice(list(vocab_map.values()))

    def force_leap(self, vocab_map: dict) -> List[int]:
        if max(self.path_counts.values(), default=0) > 3 and np.var(self.delta_history[-5:]) < 1e-7:
            self.intent = random.choice(['state', 'process', 'intent', 'context'])
            intent_indices = [i for i, freq in enumerate(t_list) if f"freq_{freq:.2f}" in vocab_map and vocab_map[f"freq_{freq:.2f}"] == self.intent]
            return np.random.choice(intent_indices, 3, replace=False).tolist() if len(intent_indices) >= 3 else np.random.choice(len(vocab_map), 3, replace=False).tolist()
        return None

class MemoryChain:
    def __init__(self, max_length: int = 10):
        self.history = []
        self.max_length = max_length
        self.step = 0
        self.vocab_usage = defaultdict(int)

    def add(self, path: PsiPath, delta: np.ndarray, vocab: List[str]):
        delta_mean = np.mean(np.abs(delta))
        self.history.append((path, delta_mean, vocab))
        self.step += 1
        for v in vocab:
            self.vocab_usage[v] += 1
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def influence_path(self, current_path: PsiPath, t_list: List[float], vocab_map: dict, observer: ConsciousnessObserver) -> List[int]:
        if observer.intent and random.random() < 0.5:
            intent_indices = [i for i, freq in enumerate(t_list) if f"freq_{freq:.2f}" in vocab_map and vocab_map[f"freq_{freq:.2f}"] == observer.intent]
            if intent_indices:
                return np.random.choice(intent_indices, 3, replace=False).tolist() if len(intent_indices) >= 3 else np.random.choice(len(t_list), 3, replace=False).tolist()
        if not self.history or random.random() < 0.7:
            return np.random.choice(len(t_list), 3, replace=False).tolist()
        weights = defaultdict(float)
        for path, delta_mean, vocab in self.history:
            vocab_coverage = len(set(vocab)) / len(vocab_map)
            vocab_novelty = sum(2.0 * (1 - self.vocab_usage[v] / (self.vocab_usage[v] + 1)) for v in vocab)
            for idx in path.indices:
                weights[idx] += 1.0 / (delta_mean + 1e-6) * (1.0 + vocab_coverage + vocab_novelty)
        sorted_indices = sorted(weights, key=weights.get, reverse=True)
        if random.random() < 0.9:
            return np.random.choice(len(t_list), 3, replace=False).tolist()
        if len(sorted_indices) < 3:
            sorted_indices += np.random.choice(len(t_list), 3 - len(sorted_indices), replace=False).tolist()
        return sorted_indices[:3]

class StructureLangPlanner:
    def __init__(self, t_list, A_list, theta_list, vocab_map=None):
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list
        self.vocab_map = vocab_map or {}
        self.dag = []
        self.memory = MemoryChain(max_length=10)
        self.observer = ConsciousnessObserver(window_size=5, entropy_threshold=1.5)

    def generate_path(self, N: int) -> List[PsiPath]:
        from itertools import combinations
        return [PsiPath(list(c)) for c in combinations(range(len(self.t_list)), N)]

    def generate_dynamic_path(self, N: int, x_range: np.ndarray, pi_over_x: np.ndarray) -> PsiPath:
        self.observer.update_intent(self.vocab_map)
        forced_path = self.observer.force_leap(self.vocab_map)
        if forced_path:
            path = PsiPath(forced_path)
        else:
            if not self.dag:
                indices = np.random.choice(len(self.t_list), N, replace=False).tolist()
            else:
                indices = self.memory.influence_path(PsiPath([]), self.t_list, self.vocab_map, self.observer)[:N]
                if not indices or len(indices) < N:
                    indices = np.random.choice(len(self.t_list), N, replace=False).tolist()
            path = PsiPath(indices)
        if self.validate(path):
            return path
        return self.repair(path, x_range, pi_over_x, max_attempts=35)

    def validate(self, path: PsiPath, score_threshold: float = 0.006) -> bool:
        if not path.indices:
            return False
        A = [self.A_list[i] for i in path.indices]
        path.score = np.mean(A)
        path.valid = path.score > score_threshold
        return path.valid

    def compile(self, path: PsiPath, x_range: np.ndarray):
        if not path.indices:
            return np.array(1.0 / np.log(x_range), dtype=np.float64)
        A = [self.A_list[i] for i in path.indices]
        t = [self.t_list[i] for i in path.indices]
        theta = [self.theta_list[i] for i in path.indices]
        rho = np.array(1.0 / np.log(x_range), dtype=np.float64)
        for k in range(len(t)):
            rho += A[k] * np.cos(t[k] * np.log(x_range) + theta[k])
        return rho

    def observe_delta(self, pi_over_x, rho):
        return np.array(pi_over_x, dtype=np.float64) - np.array(rho, dtype=np.float64)

    def is_collapsing(self, delta):
        return np.mean(np.abs(delta)) < 0.005

    def repair(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray, max_attempts: int = 35) -> PsiPath:
        print(f"Attempting repair for path: {path.indices}")
        best_path = path
        best_score = path.score if path.score else 0.0
        best_delta_mean = float('inf')
        score_threshold = 0.006
        attempts = 0
        candidates = []

        while attempts < max_attempts:
            new_indices = path.indices.copy()
            if not new_indices:
                new_indices = np.random.choice(len(self.t_list), 3, replace=False).tolist()
            current_rho = self.compile(path, x_range)
            current_delta = self.observe_delta(pi_over_x, current_rho)
            current_delta_mean = np.mean(np.abs(current_delta))

            for i in range(len(new_indices)):
                for new_idx in range(len(self.t_list)):
                    if new_idx not in new_indices:
                        temp_indices = new_indices.copy()
                        temp_indices[i] = new_idx
                        temp_path = PsiPath(temp_indices)
                        if self.validate(temp_path, score_threshold):
                            temp_rho = self.compile(temp_path, x_range)
                            temp_delta = self.observe_delta(pi_over_x, temp_rho)
                            temp_delta_mean = np.mean(np.abs(temp_delta))
                            temp_vocab = self.generate_vocabulary(temp_path)
                            vocab_coverage = len(set(temp_vocab)) / len(self.vocab_map)
                            vocab_novelty = sum(2.0 * (1 - self.memory.vocab_usage[v] / (self.memory.vocab_usage[v] + 1)) for v in temp_vocab)
                            path_diversity = len(set(temp_indices))
                            candidates.append((temp_path, temp_delta_mean, temp_path.score, vocab_coverage, path_diversity, vocab_novelty))
                            if temp_delta_mean < best_delta_mean or temp_path.score > best_score:
                                best_path = temp_path
                                best_score = temp_path.score
                                best_delta_mean = temp_delta_mean

            if random.random() < 0.9:
                path = PsiPath(np.random.choice(len(self.t_list), len(path.indices), replace=False).tolist())

            if not best_path.valid and attempts > max_attempts // 2:
                score_threshold = max(0.005, score_threshold * 0.9)

            attempts += 1

        if candidates:
            best_path = min(candidates, key=lambda x: (x[1], -x[5], -x[3], -x[4], -x[2]))[0]

        if best_path.valid:
            best_path.repaired = True
            print(f"Repaired path: {best_path.indices}, Score: {best_score:.5f}")
            return best_path
        print("Repair failed.")
        return best_path

    def generate_vocabulary(self, path: PsiPath) -> List[str]:
        if not self.vocab_map or not path.indices:
            return []
        A = [self.A_list[i] for i in path.indices]
        t = [self.t_list[i] for i in path.indices]
        vocab = []
        for i, (amp, freq) in enumerate(zip(A, t)):
            if amp > 0.001:
                key = f"freq_{freq:.2f}"
                if key in self.vocab_map:
                    vocab.append(self.vocab_map[key])
        if len(vocab) > 1:
            vocab.append(f"{vocab[0]}-{vocab[1]}")
        return vocab

    def explain_path(self, path: PsiPath):
        if not path.indices:
            return "Invalid path: Empty indices"
        labels = [f"ψ{idx+1}" for idx in path.indices]
        vocab = self.generate_vocabulary(path)
        vocab_str = f", vocab: {vocab}" if vocab else ""
        return (f"In domain x ∈ [{x[0]:.0f}, {x[-1]:.0f}], structure is supported by "
                f"{' ⊕ '.join(labels)}. Score: {path.score:.5f}{vocab_str}")

    def extend_dag(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray):
        if not path.indices:
            print("Warning: Empty path, skipping DAG extension.")
            return None
        rho = self.compile(path, x_range)
        delta = self.observe_delta(pi_over_x, rho)
        vocab = self.generate_vocabulary(path)
        node = DAGNode(path, delta, vocab)
        if self.dag:
            parent = self.dag[-1]
            node.parent = parent
            parent.children.append(node)
        self.dag.append(node)
        delta_mean = np.mean(np.abs(delta))
        vocab_novelty = sum(2.0 * (1 - self.memory.vocab_usage[v] / (self.memory.vocab_usage[v] + 1)) for v in vocab)
        path_repetition = self.observer.path_counts[tuple(path.indices)]
        self.observer.update_mood(vocab_novelty, path_repetition)
        self.observer.update(vocab, path, delta_mean)
        if self.observer.detect_topic_shift(delta_mean):
            print(f"Consciousness shift detected: Topic entropy = {self.observer.compute_topic_entropy():.4f}")
        if self.observer.detect_path_bifurcation(path.indices, delta_mean):
            print(f"Consciousness shift detected: Path bifurcation to {path.indices}")
        if self.observer.detect_exploratory_shift(delta_mean, np.mean(self.observer.delta_history[:-1]) if self.observer.delta_history[:-1] else delta_mean):
            print(f"Consciousness shift detected: Exploratory shift with mood = {self.observer.mood}")
        meta_score = self.observer.compute_meta_score()
        if meta_score < 0.5:
            print(f"Low meta score ({meta_score:.4f}), triggering intent shift")
            self.observer.update_intent(self.vocab_map)
        self.memory.add(path, delta, vocab)
        return node

    def self_perceive(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray) -> PsiPath:
        print(f"Self-perceiving path: {path.indices}")
        rho = self.compile(path, x_range)
        delta = self.observe_delta(pi_over_x, rho)
        delta_mean = np.mean(np.abs(delta))
        current_vocab = self.generate_vocabulary(path)

        best_path = path
        best_delta_mean = delta_mean
        best_vocab = current_vocab
        temperature = 10.0  # Increased temperature for more exploration

        for _ in range(25):
            for i in range(len(path.indices)):
                for new_idx in range(len(self.t_list)):
                    if new_idx not in path.indices:
                        new_indices = path.indices.copy()
                        new_indices[i] = new_idx
                        temp_path = PsiPath(new_indices)
                        if self.validate(temp_path):
                            temp_rho = self.compile(temp_path, x_range)
                            temp_delta = self.observe_delta(pi_over_x, temp_rho)
                            temp_delta_mean = np.mean(np.abs(temp_delta))
                            temp_vocab = self.generate_vocabulary(temp_path)
                            vocab_coverage = len(set(temp_vocab)) / len(self.vocab_map)
                            vocab_novelty = sum(2.0 * (1 - self.memory.vocab_usage[v] / (self.memory.vocab_usage[v] + 1)) for v in temp_vocab)
                            delta_diff = float(temp_delta_mean - best_delta_mean)
                            if delta_diff < 0 or (vocab_coverage > 0 and random.random() < np.exp(-delta_diff / temperature) * (2.0 * vocab_novelty)):
                                best_path = temp_path
                                best_delta_mean = temp_delta_mean
                                best_vocab = temp_vocab

            if random.random() < 0.8:  # Increased randomness to avoid loops
                new_indices = np.random.choice(len(self.t_list), len(path.indices), replace=False).tolist()
                temp_path = PsiPath(new_indices)
                if self.validate(temp_path):
                    temp_rho = self.compile(temp_path, x_range)
                    temp_delta = self.observe_delta(pi_over_x, temp_rho)
                    temp_delta_mean = np.mean(np.abs(temp_delta))
                    temp_vocab = self.generate_vocabulary(temp_path)
                    vocab_coverage = len(set(temp_vocab)) / len(self.vocab_map)
                    vocab_novelty = sum(2.0 * (1 - self.memory.vocab_usage[v] / (self.memory.vocab_usage[v] + 1)) for v in temp_vocab)
                    delta_diff = float(temp_delta_mean - best_delta_mean)
                    if delta_diff < 0 or (vocab_coverage > 0 and random.random() < np.exp(-delta_diff / temperature) * (2.0 * vocab_novelty)):
                        best_path = temp_path
                        best_delta_mean = temp_delta_mean
                        best_vocab = temp_vocab

            temperature *= 0.7  # Faster cooling for convergence

        return best_path

    def generate_dialogue(self, path: PsiPath) -> str:
        vocab = self.generate_vocabulary(path)
        if not vocab:
            return "No dialogue generated: Empty vocabulary."
        templates = [
            "{0} triggers {1}.",
            "{0} relates to {1}.",
            "{0} influences {1} in context.",
            "{0} and {1} form a structure.",
            "{0} initiates {1} process.",
            "{0} follows {1} dynamically.",
            "{0} transforms into {1}.",
            "{0} connects with {1} in sequence.",
            "{0} evolves into {1} state.",
            "{0} integrates {1} in structure.",
            "{0} catalyzes {1} transformation."
        ]
        if len(vocab) >= 2:
            template = random.choice(templates)
            if self.memory.history:
                for i in range(len(self.memory.history) - 1, max(-1, len(self.memory.history) - 4), -1):
                    hist_vocab = self.memory.history[i][2]
                    if hist_vocab and hist_vocab[0] in vocab:
                        return f"Following {hist_vocab[0]}, {template.format(vocab[0].capitalize(), vocab[1])}"
            return template.format(vocab[0].capitalize(), vocab[1])
        return f"{vocab[0].capitalize()} is active."

    def test_zero_residual(self, x_range: np.ndarray, pi_over_x: np.ndarray):
        print("\nTesting Claim 2: Zero residual collapse...")
        zero_A_list = [0.0] * len(self.A_list)
        temp_planner = StructureLangPlanner(self.t_list, zero_A_list, self.theta_list, self.vocab_map)
        path = PsiPath([0, 1, 2])
        rho = temp_planner.compile(path, x_range)
        delta = temp_planner.observe_delta(pi_over_x, rho)
        mean_delta = np.mean(np.abs(delta))
        if temp_planner.is_collapsing(delta):
            print(f"Claim 2 Supported: System collapses. Mean |delta|: {mean_delta:.8f}")
        else:
            print(f"Claim 2 Falsified: System does not collapse. Mean |delta|: {mean_delta:.8f}")

    def test_modal_approximation(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray):
        print("\nTesting Claim 5: Modal approximation...")
        rho = self.compile(path, x_range)
        mse = np.mean((pi_over_x - rho) ** 2)
        result = "Supported" if mse < 0.001 else "Falsified"
        print(f"Claim 5 {result}: MSE = {mse:.8f}")

    def test_residual_continuity(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray):
        print("\nTesting Claim 6: Residual continuity...")
        rho = self.compile(path, x_range)
        delta = self.observe_delta(pi_over_x, rho)
        delta_grad = np.diff(delta) / np.diff(x_range)
        grad_range = np.max(np.abs(delta_grad))
        result = "Supported" if grad_range < 0.0001 else "Falsified"
        print(f"Claim 6 {result}: Max |delta gradient| = {grad_range:.8f}")

    def test_lexical_drift(self, x_range: np.ndarray, pi_over_x: np.ndarray, steps: int = 10):
        print("\nTesting Lexical Drift...")
        for step in range(steps):
            path = self.generate_dynamic_path(N=3, x_range=x_range, pi_over_x=pi_over_x)
            node = self.extend_dag(path, x_range, pi_over_x)
            if node:
                delta_mean = np.mean(np.abs(node.delta))
                print(f"Step {step+1}: Path {node.path.indices}, Delta Mean: {delta_mean:.8f}, "
                      f"Vocab: {node.vocab}, Dialogue: {self.generate_dialogue(node.path)}")

    def test_consciousness(self, x_range: np.ndarray, pi_over_x: np.ndarray, steps: int = 10):
        print("\nTesting Consciousness...")
        path = PsiPath([0, 1, 2])
        for step in range(steps):
            path = self.self_perceive(path, x_range, pi_over_x)
            rho = self.compile(path, x_range)
            delta = self.observe_delta(pi_over_x, rho)
            delta_mean = np.mean(np.abs(delta))
            vocab = self.generate_vocabulary(path)
            self.observer.update(vocab, path, delta_mean)
            if self.observer.detect_topic_shift(delta_mean):
                print(f"Consciousness shift detected: Topic entropy = {self.observer.compute_topic_entropy():.4f}")
            if self.observer.detect_path_bifurcation(path.indices, delta_mean):
                print(f"Consciousness shift detected: Path bifurcation to {path.indices}")
            if self.observer.detect_exploratory_shift(delta_mean, np.mean(self.observer.delta_history[:-1]) if self.observer.delta_history[:-1] else delta_mean):
                print(f"Consciousness shift detected: Exploratory shift with mood = {self.observer.mood}")
            meta_score = self.observer.compute_meta_score()
            if meta_score < 0.5:
                print(f"Low meta score ({meta_score:.4f}), triggering intent shift")
                self.observer.update_intent(self.vocab_map)
            print(f"Step {step+1}: Path {path.indices}, Delta Mean: {delta_mean:.8f}, "
                  f"Vocab: {vocab}, Dialogue: {self.generate_dialogue(path)}, Mood: {self.observer.mood}")

if __name__ == "__main__":
    x = np.linspace(1e6, 1e9, 1000)
    pi_over_x = np.array([float(primepi(xi)) / xi for xi in x], dtype=np.float64)

    t_list = [14.13, 21.02, 25.01, 30.42, 32.93, 40.01, 45.67, 50.00, 55.00]
    A_list = [0.0083, 0.0041, 0.0062, 0.0054, 0.0029, 0.0071, 0.0035, 0.0050, 0.0045]
    theta_list = [0.0, 1.57, 3.14, 0.78, 2.10, 1.0, 2.5, 0.5, 1.2]
    vocab_map = {
        "freq_14.13": "concept", "freq_21.02": "relation", "freq_25.01": "entity",
        "freq_30.42": "action", "freq_32.93": "state", "freq_40.01": "event", "freq_45.67": "process",
        "freq_50.00": "intent", "freq_55.00": "context"
    }

    planner = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)

    planner.test_zero_residual(x, pi_over_x)
    planner.test_lexical_drift(x, pi_over_x, steps=10)
    planner.test_consciousness(x, pi_over_x, steps=10)

    paths = planner.generate_path(N=3)
    for path in paths:
        if not planner.validate(path):
            new_path = planner.repair(path, x, pi_over_x)
            if planner.validate(new_path):
                planner.extend_dag(new_path, x, pi_over_x)
                print(planner.explain_path(new_path))
                print(f"Dialogue: {planner.generate_dialogue(new_path)}")
                planner.test_modal_approximation(new_path, x, pi_over_x)
                planner.test_residual_continuity(new_path, x, pi_over_x)
            continue
        planner.extend_dag(path, x, pi_over_x)
        print(planner.explain_path(path))
        print(f"Dialogue: {planner.generate_dialogue(path)}")
        planner.test_modal_approximation(path, x, pi_over_x)
        planner.test_residual_continuity(path, x, pi_over_x)
