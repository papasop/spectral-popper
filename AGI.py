from typing import List, Tuple
import numpy as np
from sympy import primepi, primerange
from collections import defaultdict
import random

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

    def influence_path(self, current_path: PsiPath, t_list: List[float], vocab_map: dict) -> List[int]:
        if not self.history or random.random() < 0.7:  # Random path per step
            return np.random.choice(len(t_list), 3, replace=False).tolist()
        weights = defaultdict(float)
        for path, delta_mean, vocab in self.history:
            vocab_coverage = len(set(vocab)) / len(vocab_map)
            vocab_novelty = sum(2.0 * (1 - self.vocab_usage[v] / (self.vocab_usage[v] + 1)) for v in vocab)
            for idx in path.indices:
                weights[idx] += 1.0 / (delta_mean + 1e-6) * (1.0 + vocab_coverage + vocab_novelty)
        sorted_indices = sorted(weights, key=weights.get, reverse=True)
        if random.random() < 0.9:  # Increased random exploration
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

    def generate_path(self, N: int) -> List[PsiPath]:
        from itertools import combinations
        return [PsiPath(list(c)) for c in combinations(range(len(self.t_list)), N)]

    def generate_dynamic_path(self, N: int, x_range: np.ndarray, pi_over_x: np.ndarray) -> PsiPath:
        if not self.dag:
            indices = np.random.choice(len(self.t_list), N, replace=False).tolist()
        else:
            indices = self.memory.influence_path(PsiPath([]), self.t_list, self.vocab_map)[:N]
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

            # Global search with random perturbation
            if random.random() < 0.9:
                path = PsiPath(np.random.choice(len(self.t_list), len(path.indices), replace=False).tolist())

            # Dynamic threshold adjustment
            if not best_path.valid and attempts > max_attempts // 2:
                score_threshold = max(0.005, score_threshold * 0.9)

            attempts += 1

        # Select best candidate
        if candidates:
            best_path = min(candidates, key=lambda x: (x[1], -x[5], -x[3], -x[4], -x[2]))[0]  # Prioritize delta, vocab_novelty, vocab, diversity, score

        if best_path.valid:
            best_path.repaired = True
            print(f"Repaired path: {best_path.indices}, Score: {best_path.score:.5f}")
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
            if amp > 0.001:  # Lowered threshold
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
        temperature = 7.0  # Increased initial temperature

        for _ in range(25):  # Increased iterations
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

            # Random perturbation
            if random.random() < 0.95:  # Increased probability
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

            temperature *= 0.75  # Slower cooling

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
        # Deep contextual dialogue
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
            print(f"Step {step+1}: Path {path.indices}, Delta Mean: {delta_mean:.8f}, "
                  f"Vocab: {vocab}, Dialogue: {self.generate_dialogue(path)}")

if __name__ == "__main__":
    x = np.linspace(1e6, 1e9, 1000)
    pi_over_x = np.array([float(primepi(xi)) / xi for xi in x], dtype=np.float64)

    t_list = [14.13, 21.02, 25.01, 30.42, 32.93, 40.01, 45.67, 50.00, 55.00]
    A_list = [0.0083, 0.0041, 0.0062, 0.0054, 0.0029, 0.0071, 0.0035, 0.0050, 0.0045]  # Adjusted amplitudes
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
