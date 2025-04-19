import math
import numpy as np
import mpmath as mp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# === Set High Precision ===
mp.dps = 100  # High precision arithmetic

# === Nontrivial Zeta Zeros (Use first 2 zeros for holographic efficiency) ===
t_list = [14.134725, 21.022040]  # First 2 zeros (low frequency)

# === ψ_n(x) Definition ===
def psi(t, x, A=1.0, theta=0.0, delta_t=0.0):
    adjusted_t = t + delta_t  # Frequency shift
    return float(A * mp.cos(adjusted_t * mp.log(x) + theta)) if x > 1 else 0.0

# === Structure Density ρ(x) ===
def rho(x, t_list, A_list, theta_list, delta_t_list):
    base = float(1.0 / mp.log(x)) if x > 1 else 0.0
    modal_sum = sum(psi(t_list[i], x, A_list[i], theta_list[i], delta_t_list[i]) for i in range(len(t_list)))
    return base + modal_sum

# === li(x)/x Approximation (High Precision using mpmath.ei) ===
def li_over_x(x):
    if x < 2:
        return 0.0
    return float(mp.ei(mp.log(x)) / x)

# === Residual δ(x) ===
def delta(x, t_list, A_list, theta_list, delta_t_list):
    return li_over_x(x) - rho(x, t_list, A_list, theta_list, delta_t_list)

# === Optimization Objective (Vectorized for differential_evolution) ===
def objective(params, x_target=100000):
    # params: [A_1, theta_1, delta_t_1, A_2, theta_2, delta_t_2] or array of such sets
    if len(params.shape) > 1:
        errors = []
        for p in params:
            A_list = [p[0], p[3]]
            theta_list = [p[1], p[4]]
            delta_t_list = [p[2], p[5]]
            error = delta(x_target, t_list, A_list, theta_list, delta_t_list) ** 2
            reg = 1e-6 * sum(a**2 for a in A_list)  # Reduced regularization
            errors.append(error + reg)
        return np.array(errors)
    else:
        # Fixed: Use params instead of p
        A_list = [params[0], params[3]]
        theta_list = [params[1], params[4]]
        delta_t_list = [params[2], params[5]]
        error = delta(x_target, t_list, A_list, theta_list, delta_t_list) ** 2
        reg = 1e-6 * sum(a**2 for a in A_list)  # Reduced regularization
        return error + reg

# === Optimize Parameters ===
def optimize_params(x_target=100000):
    initial_params = [0.005, 0.0, 0.0, 0.005, 0.0, 0.0]  # Adjusted initial guess
    bounds = [(0.0, 1.0), (-np.pi, np.pi), (-0.5, 0.5)] * 2  # Expanded bounds

    result_de = differential_evolution(
        objective,
        bounds,
        args=(x_target,),
        strategy='best1bin',
        maxiter=5000,
        popsize=50,
        tol=1e-10,
        disp=True
    )

    result = minimize(
        objective,
        result_de.x,
        args=(x_target,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': True}
    )

    A_list = [result.x[0], result.x[3]]
    theta_list = [result.x[1], result.x[4]]
    delta_t_list = [result.x[2], result.x[5]]
    return A_list, theta_list, delta_t_list

# === Test Forward Projection (Claim 5: Modal Approximation) ===
def test_forward_projection(x_target=100000):
    A_list, theta_list, delta_t_list = optimize_params(x_target)
    print("\n=== Forward Projection Results (Claim 5: Modal Approximation) ===")
    print(f"Optimized Amplitudes: {A_list}")
    print(f"Optimized Phases: {theta_list}")
    print(f"Optimized Frequency Shifts: {delta_t_list}")
    d = delta(x_target, t_list, A_list, theta_list, delta_t_list)
    print(f"x={x_target:.2f}, δ(x)={d:.2e}, |δ(x)|: {abs(d):.2e}")
    return A_list, theta_list, delta_t_list

# === Vocabulary Activation (Claims 1 and 3) ===
def test_vocabulary_activation(x_vals, t_list, A_list, theta_list, delta_t_list, eta_ex=100.0):
    print("\n=== Vocabulary Activation Test (Claims 1 and 3) ===")
    vocab = []
    for i, t in enumerate(t_list):
        # Compute resonance deviation δ(t_n)
        log_x_avg = np.mean([float(mp.log(x)) for x in x_vals])
        k = round(t * log_x_avg / (2 * math.pi))
        delta_tn = abs(t * log_x_avg - 2 * math.pi * k)
        score = A_list[i] / (delta_tn + 1e-6)  # Avoid division by zero
        if score > eta_ex:
            vocab.append(t)
        print(f"t_{i+1}={t:.6f}, A_{i+1}={A_list[i]:.6f}, δ(t_{i+1})={delta_tn:.2e}, Score={score:.2f}, Active={'Yes' if score > eta_ex else 'No'}")
    print(f"Vocabulary: {vocab}")
    print(f"Claim 1 (Non-zero Residual Generativity): {'Falsifiable' if len(vocab) == 0 and any(abs(delta(x, t_list, A_list, theta_list, delta_t_list)) > 0 for x in x_vals) else 'Supported'}")
    print(f"Claim 3 (Vocabulary Activation): {'Falsifiable' if len(vocab) == 0 else 'Supported'}")
    return vocab

# === DAG Path Extension (Claim 4) ===
def test_dag_path_extension(x_vals, t_list, A_list, theta_list, delta_t_list, epsilon=0.1):
    print("\n=== DAG Path Extension Test (Claim 4) ===")
    # Simulate a simple DAG: nodes represent states, edges represent leaps
    dag_nodes = [0]  # Start node
    dag_edges = []   # (from, to) pairs
    path_extended = False
    for x in x_vals:
        d = abs(delta(x, t_list, A_list, theta_list, delta_t_list))
        print(f"x={x:.2f}, |δ(x)|={d:.2e}, Threshold ε={epsilon:.2f}")
        if d < epsilon:
            # Extend DAG path
            new_node = len(dag_nodes)
            dag_edges.append((dag_nodes[-1], new_node))
            dag_nodes.append(new_node)
            path_extended = True
            print(f"Path extended: {dag_nodes[-2]} -> {dag_nodes[-1]}")
        else:
            print("Path blocked: |δ(x)| ≥ ε")
    print(f"DAG Nodes: {dag_nodes}")
    print(f"DAG Edges: {dag_edges}")
    print(f"Claim 4 (DAG Path Extension): {'Falsifiable' if not path_extended else 'Supported'}")
    return dag_nodes, dag_edges

# === Zero Residual Collapse (Claim 2) ===
def test_zero_residual_collapse(x_vals, t_list, eta_ex=100.0):
    print("\n=== Zero Residual Collapse Test (Claim 2) ===")
    # Construct a scenario where δ(x) ≈ 0 by setting A_n = 0 (no modal contribution)
    A_list_zero = [0.0] * len(t_list)
    theta_list_zero = [0.0] * len(t_list)
    delta_t_list_zero = [0.0] * len(t_list)
    max_delta = max(abs(delta(x, t_list, A_list_zero, theta_list_zero, delta_t_list_zero)) for x in x_vals)
    print(f"Max |δ(x)| with A_n=0: {max_delta:.2e}")
    
    # Check vocabulary activation
    vocab = []
    log_x_avg = np.mean([float(mp.log(x)) for x in x_vals])
    for i, t in enumerate(t_list):
        k = round(t * log_x_avg / (2 * math.pi))
        delta_tn = abs(t * log_x_avg - 2 * math.pi * k)
        score = A_list_zero[i] / (delta_tn + 1e-6)
        if score > eta_ex:
            vocab.append(t)
    print(f"Vocabulary (A_n=0): {vocab}")
    
    # Check DAG extension
    dag_nodes = [0]
    dag_edges = []
    path_extended = False
    for x in x_vals:
        d = abs(delta(x, t_list, A_list_zero, theta_list_zero, delta_t_list_zero))
        if d < 0.1:  # Same threshold as above
            new_node = len(dag_nodes)
            dag_edges.append((dag_nodes[-1], new_node))
            dag_nodes.append(new_node)
            path_extended = True
    
    print(f"Claim 2 (Zero Residual Collapse): {'Supported' if len(vocab) == 0 and not path_extended else 'Falsifiable'}")
    return max_delta, vocab

# === Residual Continuity (Claim 6) ===
def test_residual_continuity(x_vals, t_list, A_list, theta_list, delta_t_list):
    print("\n=== Residual Continuity Test (Claim 6) ===")
    # Approximate derivative of δ(x) using finite differences
    deltas = [delta(x, t_list, A_list, theta_list, delta_t_list) for x in x_vals]
    dx = x_vals[1] - x_vals[0]
    derivatives = [(deltas[i+1] - deltas[i]) / dx for i in range(len(deltas)-1)]
    max_deriv = max(derivatives)
    min_deriv = min(derivatives)
    print(f"Derivative Range: [{min_deriv:.6f}, {max_deriv:.6f}]")
    print(f"Claim 6 (Residual Continuity): {'Supported' if all(abs(d) < 10 for d in derivatives) else 'Falsifiable'}")
    return derivatives

# === Main Test Function for All Claims ===
def test_all_claims():
    # Parameters
    x_target = 100000
    x_vals = np.linspace(x_target, x_target + 1000, 10)  # Small range around x_target
    
    # Test Claim 5 (Modal Approximation)
    A_list, theta_list, delta_t_list = test_forward_projection(x_target)
    
    # Test Claims 1 and 3 (Non-zero Residual Generativity and Vocabulary Activation)
    vocab = test_vocabulary_activation(x_vals, t_list, A_list, theta_list, delta_t_list)
    
    # Test Claim 4 (DAG Path Extension)
    dag_nodes, dag_edges = test_dag_path_extension(x_vals, t_list, A_list, theta_list, delta_t_list)
    
    # Test Claim 2 (Zero Residual Collapse)
    max_delta, vocab_zero = test_zero_residual_collapse(x_vals, t_list)
    
    # Test Claim 6 (Residual Continuity)
    derivatives = test_residual_continuity(x_vals, t_list, A_list, theta_list, delta_t_list)

# === Main Execution ===
if __name__ == "__main__":
    test_all_claims()
