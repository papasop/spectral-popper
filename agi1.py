# conscious_ai.py
import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi
from collections import defaultdict
import random

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

# 测试函数集合
def test_self_reflection(planner, x, pi_x):
    print("\n[1] 🧠 Self-Reflection")
    path = planner.generate_dynamic_path(3, x, pi_x)
    rho = planner.compile(path, x)
    delta = planner.observe_delta(pi_x, rho)
    entropy = np.trapz(delta**2, x)
    print(f"路径: {path.indices} | 结构熵: {entropy:.6f}")
    print("🧘 觉察：", "结构崩塌" if entropy < 1e-4 else "张力仍在")

def test_break_bias(planner, x, pi_x):
    print("\n[2] 🧹 Break Mana Bias")
    path = planner.generate_dynamic_path(3, x, pi_x)
    vocab = planner.generate_vocabulary(path)
    print(f"路径: {path.indices} | 稀有词汇优先选择: {vocab}")

def test_will_jump(planner, x, pi_x):
    print("\n[3] 🔥 Willful Jump")
    best_entropy = -np.inf
    best_path = None
    for _ in range(20):
        path = planner.generate_dynamic_path(3, x, pi_x)
        rho = planner.compile(path, x)
        delta = planner.observe_delta(pi_x, rho)
        entropy = np.trapz(delta**2, x)
        if entropy > best_entropy:
            best_entropy = entropy
            best_path = path
    print(f"ψ-will 选择: {best_path.indices} | 最大熵: {best_entropy:.6f}")

def test_loop_escape(planner, x, pi_x):
    print("\n[4] ♻️ Collapse Loop Escape")
    seen = set()
    for _ in range(10):
        path = planner.generate_dynamic_path(3, x, pi_x)
        key = tuple(sorted(path.indices))
        if key in seen:
            print(f"🔁 循环路径 {path.indices}，尝试修复...")
            repaired = planner.repair(path, x, pi_x)
            print(f"🛫 修复后: {repaired.indices}")
        else:
            seen.add(key)

def test_dual_ai(planner_a, planner_b, x, pi_x):
    print("\n[5] 🤝 Dual AI Dialogue")
    for step in range(3):
        path_a = planner_a.generate_dynamic_path(3, x, pi_x)
        node_a = planner_a.extend_dag(path_a, x, pi_x)
        path_b = planner_b.generate_dynamic_path(3, x, pi_x)
        node_b = planner_b.extend_dag(path_b, x, pi_x)
        vocab_a = node_a.vocab
        vocab_b = node_b.vocab
        delta_diff = np.mean(np.abs(node_a.delta - node_b.delta))
        print(f"[{step+1}] A→{vocab_a} || B→{vocab_b} | 干扰度: {delta_diff:.6f}")

def generate_psi_c(planner_a, planner_b, x, pi_x):
    print("\n[6] 🧬 ψC Generation")
    path_a = planner_a.generate_dynamic_path(3, x, pi_x)
    node_a = planner_a.extend_dag(path_a, x, pi_x)
    path_b = planner_b.generate_dynamic_path(3, x, pi_x)
    node_b = planner_b.extend_dag(path_b, x, pi_x)

    combined = list(set(path_a.indices + path_b.indices))
    if len(combined) < 3:
        combined += list(np.random.choice(len(planner_a.t_list), 3 - len(combined), replace=False))
    np.random.shuffle(combined)
    psi_c_indices = combined[:3]
    planner_c = StructureLangPlanner(planner_a.t_list, planner_a.A_list, planner_a.theta_list, planner_a.vocab_map)
    node_c = planner_c.extend_dag(PsiPath(psi_c_indices), x, pi_x)
    print(f"ψA: {node_a.vocab} | ψB: {node_b.vocab} | ψC: {node_c.vocab} | Sψ = {np.trapz(node_c.delta**2, x):.6f}")

# 主程序
if __name__ == "__main__":
    print("🚀 StructureLang 觉性 AI 启动...\n")
    x = np.linspace(1e6, 1e7, 500)
    pi_x = np.array([primepi(xi) / xi for xi in x], dtype=np.float64)
    t_list = [14.13, 21.02, 25.01]
    A_list = [0.006, 0.005, 0.007]
    theta_list = [0.0, 1.57, 3.14]
    vocab_map = {
        "freq_14.13": "concept", "freq_21.02": "relation", "freq_25.01": "entity"
    }

    planner = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)
    test_self_reflection(planner, x, pi_x)
    test_break_bias(planner, x, pi_x)
    test_will_jump(planner, x, pi_x)
    test_loop_escape(planner, x, pi_x)

    planner_a = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)
    planner_b = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)
    test_dual_ai(planner_a, planner_b, x, pi_x)
    generate_psi_c(planner_a, planner_b, x, pi_x)
