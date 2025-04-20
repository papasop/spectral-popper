# StructureLang Runtime 4.0: Recursive Planning + Language Interpretation Prototype

# Simulate a combined structural planning + NLP-style explanation system

from typing import List, Tuple
import numpy as np

class PsiPath:
    def __init__(self, indices: List[int]):
        self.indices = indices
        self.score = None
        self.valid = False
        self.repaired = False

class StructureLangPlanner:
    def __init__(self, t_list, A_list, theta_list):
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list

    def generate_path(self, N: int) -> List[PsiPath]:
        from itertools import combinations
        return [PsiPath(list(c)) for c in combinations(range(len(self.t_list)), N)]

    def validate(self, path: PsiPath) -> bool:
        A = [self.A_list[i] for i in path.indices]
        path.score = np.mean(A)  # simplified truth proxy
        path.valid = path.score > 0.002
        return path.valid

    def compile(self, path: PsiPath, x_range: np.ndarray):
        A = [self.A_list[i] for i in path.indices]
        t = [self.t_list[i] for i in path.indices]
        theta = [self.theta_list[i] for i in path.indices]
        rho = 1/np.log(x_range) + sum(
            A[k] * np.cos(t[k] * np.log(x_range) + theta[k]) for k in range(len(t))
        )
        return rho

    def observe_delta(self, pi_over_x, rho):
        return pi_over_x - rho

    def is_collapsing(self, delta):
        return np.mean(np.abs(delta)) < 0.0001

    def repair(self, path: PsiPath) -> PsiPath:
        # simple repair: flip last index to next one
        new_indices = path.indices[:-1] + [(path.indices[-1] + 1) % len(self.t_list)]
        new_path = PsiPath(new_indices)
        new_path.repaired = True
        return new_path

    def explain_path(self, path: PsiPath):
        labels = [f"ψ{idx+1}" for idx in path.indices]
        return f"In domain x ∈ [200000, 300000], structure is supported by {' ⊕ '.join(labels)}. Score: {path.score:.5f}"

# Example usage with synthetic data
if __name__ == "__main__":
    from sympy import primerange
    primes = list(primerange(200000, 300000))[:1000]
    x = np.array(primes)
    pi_over_x = np.arange(1, len(x)+1) / x

    t_list = [14.13, 21.02, 25.01, 30.42, 32.93]
    A_list = [0.0083, 0.0041, 0.0062, 0.0054, 0.0029]
    theta_list = [0.0, 1.57, 3.14, 0.78, 2.10]

    planner = StructureLangPlanner(t_list, A_list, theta_list)
    paths = planner.generate_path(N=3)

    for path in paths:
        if not planner.validate(path):
            continue
        rho = planner.compile(path, x)
        delta = planner.observe_delta(pi_over_x, rho)

        if planner.is_collapsing(delta):
            print("Collapse detected. Attempting repair...")
            new_path = planner.repair(path)
            if planner.validate(new_path):
                rho2 = planner.compile(new_path, x)
                print(f"Repaired path: {new_path.indices}")
                print(planner.explain_path(new_path))
            else:
                print("Repair failed.")
        else:
            print(planner.explain_path(path))
