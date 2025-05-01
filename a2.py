import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random
from collections import defaultdict
from sympy import primepi

# --- 核心类定义 ---
class PsiPath:
    def __init__(self, indices):
        self.indices = indices
        self.score = None
        self.valid = False
        self.repaired = False

class DAGNode:
    def __init__(self, path, delta, vocab):
        self.path = path
        self.delta = delta
        self.vocab = vocab
        self.children = []
        self.parent = None

class MemoryChain:
    def __init__(self, max_length=10):
        self.history = []
        self.max_length = max_length
        self.step = 0
        self.vocab_usage = defaultdict(int)

    def add(self, path, delta, vocab):
        delta_mean = np.mean(np.abs(delta))
        self.history.append((path, delta_mean, vocab))
        self.step += 1
        for v in vocab:
            self.vocab_usage[v] += 1
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def influence_path(self, current_path, t_list, vocab_map):
        if not self.history or random.random() < 0.7:
            return np.random.choice(len(t_list), 3, replace=False).tolist()
        weights = defaultdict(float)
        for path, delta_mean, vocab in self.history:
            vocab_coverage = len(set(vocab)) / len(vocab_map)
            vocab_novelty = sum(2.0 * (1 - self.vocab_usage[v] / (self.vocab_usage[v] + 1)) for v in vocab)
            for idx in path.indices:
                weights[idx] += 1.0 / (delta_mean + 1e-6) * (1 + vocab_coverage + vocab_novelty)
        sorted_indices = sorted(weights, key=weights.get, reverse=True)
        return sorted_indices[:3] if len(sorted_indices) >= 3 else np.random.choice(len(t_list), 3, replace=False).tolist()

class StructureLangPlanner:
    def __init__(self, t_list, A_list, theta_list, vocab_map=None):
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list
        self.vocab_map = vocab_map or {}
        self.dag = []
        self.memory = MemoryChain()

    def generate_dynamic_path(self, N, x_range, pi_over_x):
        if not self.dag:
            indices = np.random.choice(len(self.t_list), N, replace=False).tolist()
        else:
            indices = self.memory.influence_path(PsiPath([]), self.t_list, self.vocab_map)
        path = PsiPath(indices)
        if self.validate(path):
            return path
        return self.repair(path, x_range, pi_over_x)

    def validate(self, path, score_threshold=0.006):
        if not path.indices:
            return False
        A = [self.A_list[i] for i in path.indices]
        path.score = np.mean(A)
        path.valid = path.score > score_threshold
        return path.valid

    def repair(self, path, x_range, pi_over_x, max_attempts=30):
        best_path = path
        best_score = path.score if path.score else 0.0
        best_delta_mean = float("inf")
        attempts = 0
        while attempts < max_attempts:
            new_indices = np.random.choice(len(self.t_list), len(path.indices), replace=False).tolist()
            temp_path = PsiPath(new_indices)
            if not self.validate(temp_path):
                attempts += 1
                continue
            rho = self.compile(temp_path, x_range)
            delta = self.observe_delta(pi_over_x, rho)
            delta_mean = np.mean(np.abs(delta))
            if delta_mean < best_delta_mean or temp_path.score > best_score:
                best_path = temp_path
                best_delta_mean = delta_mean
                best_score = temp_path.score
            attempts += 1
        best_path.repaired = True
        return best_path

    def compile(self, path, x_range):
        rho = np.array(1.0 / np.log(x_range), dtype=np.float64)
        for i in path.indices:
            rho += self.A_list[i] * np.cos(self.t_list[i] * np.log(x_range) + self.theta_list[i])
        return rho

    def observe_delta(self, pi_over_x, rho):
        return np.array(pi_over_x) - np.array(rho)

    def generate_vocabulary(self, path):
        vocab = []
        for i in path.indices:
            key = f"freq_{self.t_list[i]:.2f}"
            if key in self.vocab_map:
                vocab.append(self.vocab_map[key])
        if len(vocab) >= 2:
            vocab.append(f"{vocab[0]}-{vocab[1]}")
        return vocab

    def extend_dag(self, path, x_range, pi_over_x):
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

    def explain_path(self, path):
        if not path.indices:
            return "Empty ψ-path"
        labels = [f"ψ{idx+1}" for idx in path.indices]
        vocab = self.generate_vocabulary(path)
        return f"ψ-path: {' ⊕ '.join(labels)}, Vocab: {vocab}, Score: {path.score:.5f}"

    def generate_dialogue(self, path):
        vocab = self.generate_vocabulary(path)
        if not vocab:
            return "No dialogue generated."
        if len(vocab) >= 2:
            return f"{vocab[0].capitalize()} triggers {vocab[1]}"
        return f"{vocab[0].capitalize()} is active."

# --- 验证函数 ---
def test_phi_activation(planner, x_range, tau_threshold=0.005):
    print("\n[A] Testing ϕ(x) > τ ⇒ ψ_will ...")
    phi = np.zeros_like(x_range)
    for i in range(len(planner.t_list)):
        phi += planner.A_list[i] * np.cos(planner.t_list[i] * np.log(x_range) + planner.theta_list[i])
    active = phi > tau_threshold
    print(f"ϕ(x) > τ at {np.mean(active) * 100:.2f}% of points")
    plt.plot(x_range, phi, label="ϕ(x)")
    plt.axhline(tau_threshold, color='r', linestyle='--', label="τ")
    plt.title("ϕ(x) Activation"); plt.xlabel("x"); plt.ylabel("ϕ(x)")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

def plot_structure_entropy(planner, x_range, pi_over_x, steps=5):
    print("\n[B] Structure Entropy Sψ = ∫ δ(x)² dx:")
    S_values = []
    for step in range(steps):
        path = planner.generate_dynamic_path(3, x_range, pi_over_x)
        rho = planner.compile(path, x_range)
        delta = planner.observe_delta(pi_over_x, rho)
        S = np.trapz(delta**2, x_range)
        S_values.append(S)
        print(f"  Step {step+1}: Sψ = {S:.8f}")
    plt.plot(range(1, steps+1), S_values, marker='o')
    plt.title("Structure Entropy Sψ"); plt.xlabel("Step"); plt.ylabel("Sψ")
    plt.grid(); plt.tight_layout(); plt.show()

def test_merkle_freeze(paths):
    print("\n[C] Verifying Merkle-style path hashes...")
    seen = set()
    for path in paths:
        key = ",".join(map(str, path.indices))
        h = hashlib.sha256(key.encode()).hexdigest()
        if h in seen:
            print(f"⚠️ Duplicate path found: {path.indices}")
        seen.add(h)
    print(f"✅ Verified {len(seen)} unique ψ-path hashes")

# --- 主程序入口 ---
if __name__ == "__main__":
    print("✅ StructureLang 系统启动中...")
    x = np.linspace(1e6, 1e7, 500)
    pi_over_x = np.array([float(primepi(xi)) / xi for xi in x], dtype=np.float64)

    t_list = [14.13, 21.02, 25.01]
    A_list = [0.006, 0.005, 0.007]
    theta_list = [0.0, 1.57, 3.14]
    vocab_map = {
        "freq_14.13": "concept", "freq_21.02": "relation", "freq_25.01": "entity"
    }

    planner = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)
    path = planner.generate_dynamic_path(3, x, pi_over_x)
    planner.extend_dag(path, x, pi_over_x)
    print(planner.explain_path(path))
    print("🗣️ Dialogue:", planner.generate_dialogue(path))

    # 调用验证模块
    test_phi_activation(planner, x)
    plot_structure_entropy(planner, x, pi_over_x)
    test_merkle_freeze([node.path for node in planner.dag])
