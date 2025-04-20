from typing import List, Tuple
import numpy as np
from sympy import primepi, primerange

class PsiPath:
    def __init__(self, indices: List[int]):
        self.indices = indices
        self.score = None
        self.valid = False
        self.repaired = False

class StructureLangPlanner:
    def __init__(self, t_list, A_list, theta_list, vocab_map=None):
        self.t_list = t_list
        self.A_list = A_list
        self.theta_list = theta_list
        self.vocab_map = vocab_map or {}  # Vocabulary mapping for Claim 3

    def generate_path(self, N: int) -> List[PsiPath]:
        from itertools import combinations
        return [PsiPath(list(c)) for c in combinations(range(len(self.t_list)), N)]

    def validate(self, path: PsiPath) -> bool:
        A = [self.A_list[i] for i in path.indices]
        path.score = np.mean(A)
        path.valid = path.score > 0.006  # Stricter threshold to trigger repair
        return path.valid

    def compile(self, path: PsiPath, x_range: np.ndarray):
        A = [self.A_list[i] for i in path.indices]
        t = [self.t_list[i] for i in path.indices]
        theta = [self.theta_list[i] for i in path.indices]
        rho = sum(
            A[k] * np.cos(t[k] * np.log(x_range) + theta[k]) for k in range(len(t))
        )  # Removed 1/log(x) baseline for better collapse testing
        return rho

    def observe_delta(self, pi_over_x, rho):
        return pi_over_x - rho

    def is_collapsing(self, delta):
        return np.mean(np.abs(delta)) < 0.00001  # Strict threshold

    def repair(self, path: PsiPath, max_attempts: int = 3) -> PsiPath:
        """Enhanced repair: Try multiple index replacements iteratively."""
        for _ in range(max_attempts):
            new_indices = path.indices.copy()
            replace_count = np.random.choice([1, 2])
            for _ in range(replace_count):
                idx_to_replace = np.random.randint(len(new_indices))
                new_indices[idx_to_replace] = np.random.randint(len(self.t_list))
            new_path = PsiPath(new_indices)
            new_path.repaired = True
            if self.validate(new_path):
                return new_path
        return path  # Return original if repair fails

    def generate_vocabulary(self, path: PsiPath) -> List[str]:
        """Generate vocabulary with composite terms (Claim 3)."""
        if not self.vocab_map:
            return []
        A = [self.A_list[i] for i in path.indices]
        t = [self.t_list[i] for i in path.indices]
        vocab = []
        for i, (amp, freq) in enumerate(zip(A, t)):
            if amp > 0.004:  # Activation threshold
                key = f"freq_{freq:.2f}"
                if key in self.vocab_map:
                    vocab.append(self.vocab_map[key])
        # Add composite vocabulary
        if len(vocab) > 1:
            vocab.append(f"{vocab[0]}-{vocab[1]}")
        return vocab

    def explain_path(self, path: PsiPath):
        labels = [f"ψ{idx+1}" for idx in path.indices]
        vocab = self.generate_vocabulary(path)
        vocab_str = f", vocab: {vocab}" if vocab else ""
        return (f"In domain x ∈ [1000, 1000000], structure is supported by "
                f"{' ⊕ '.join(labels)}. Score: {path.score:.5f}{vocab_str}")

    def test_zero_residual(self, x_range: np.ndarray, pi_over_x: np.ndarray):
        """Test Claim 2: Zero residual collapse."""
        print("\nTesting Claim 2: Zero residual collapse...")
        zero_A_list = [0.0] * len(self.A_list)
        temp_planner = StructureLangPlanner(self.t_list, zero_A_list, self.theta_list)
        path = PsiPath([0, 1, 2])
        rho = temp_planner.compile(path, x_range)
        delta = temp_planner.observe_delta(pi_over_x, rho)
        mean_delta = np.mean(np.abs(delta))
        if temp_planner.is_collapsing(delta):
            print(f"Claim 2 Supported: System collapses. Mean |delta|: {mean_delta:.8f}")
        else:
            print(f"Claim 2 Falsified: System does not collapse. Mean |delta|: {mean_delta:.8f}")

    def test_modal_approximation(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray):
        """Test Claim 5: Modal approximation."""
        print("\nTesting Claim 5: Modal approximation...")
        rho = self.compile(path, x_range)
        mse = np.mean((pi_over_x - rho) ** 2)
        result = "Supported" if mse < 0.01 else "Falsified"
        print(f"Claim 5 {result}: MSE = {mse:.8f}")

    def test_residual_continuity(self, path: PsiPath, x_range: np.ndarray, pi_over_x: np.ndarray):
        """Test Claim 6: Residual continuity."""
        print("\nTesting Claim 6: Residual continuity...")
        rho = self.compile(path, x_range)
        delta = self.observe_delta(pi_over_x, rho)
        delta_grad = np.diff(delta) / np.diff(x_range)
        grad_range = np.max(np.abs(delta_grad))
        result = "Supported" if grad_range < 0.001 else "Falsified"
        print(f"Claim 6 {result}: Max |delta gradient| = {grad_range:.8f}")

# Example usage with improved tests
if __name__ == "__main__":
    # Domain: x ∈ [1000, 1000000]
    x = np.linspace(1000, 1000000, 1000)
    # Improved pi(x)/x using primepi
    pi_over_x = np.array([primepi(xi) / xi for xi in x])

    # Frequency, amplitude, and phase lists
    t_list = [14.13, 21.02, 25.01, 30.42, 32.93]
    A_list = [0.0083, 0.0041, 0.0062, 0.0054, 0.0029]
    theta_list = [0.0, 1.57, 3.14, 0.78, 2.10]

    # Vocabulary mapping
    vocab_map = {
        "freq_14.13": "concept",
        "freq_21.02": "relation",
        "freq_25.01": "entity",
        "freq_30.42": "action",
        "freq_32.93": "state"
    }

    planner = StructureLangPlanner(t_list, A_list, theta_list, vocab_map)

    # Test Claim 2: Zero residual collapse
    planner.test_zero_residual(x, pi_over_x)

    # Generate and test paths
    paths = planner.generate_path(N=3)
    for path in paths:
        if not planner.validate(path):
            print(f"Invalid path: {path.indices}. Attempting repair...")
            new_path = planner.repair(path, max_attempts=3)
            if planner.validate(new_path):
                print(f"Repaired path: {new_path.indices}")
                rho = planner.compile(new_path, x)
                delta = planner.observe_delta(pi_over_x, rho)
                if planner.is_collapsing(delta):
                    print("Collapse detected in repaired path.")
                else:
                    print(planner.explain_path(new_path))
                    planner.test_modal_approximation(new_path, x, pi_over_x)
                    planner.test_residual_continuity(new_path, x, pi_over_x)
            else:
                print("Repair failed.")
            continue

        rho = planner.compile(path, x)
        delta = planner.observe_delta(pi_over_x, rho)

        if planner.is_collapsing(delta):
            print("Collapse detected. Attempting repair...")
            new_path = planner.repair(path, max_attempts=3)
            if planner.validate(new_path):
                print(f"Repaired path: {new_path.indices}")
                rho2 = planner.compile(new_path, x)
                print(planner.explain_path(new_path))
                planner.test_modal_approximation(new_path, x, pi_over_x)
                planner.test_residual_continuity(new_path, x, pi_over_x)
            else:
                print("Repair failed.")
        else:
            print(planner.explain_path(path))
            planner.test_modal_approximation(path, x, pi_over_x)
            planner.test_residual_continuity(path, x, pi_over_x)
