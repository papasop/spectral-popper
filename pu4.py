import math
from scipy.special import expi

# === ψ_n(x) Definition ===
def psi(t, x, A=1.0, theta=0.0):
    return A * math.cos(t * math.log(x) + theta)

# === Structure Density ρ(x) ===
def rho(x, t_list, A_list, theta_list):
    base = 1.0 / math.log(x) if x > 1 else 0.0
    modal_sum = sum(A_list[i] * math.cos(t_list[i] * math.log(x) + theta_list[i]) for i in range(len(t_list)))
    return base + modal_sum

# === Improved π(x)/x approximation using li(x)/x ===
def li_over_x(x):
    if x < 2:
        return 0.0
    li_x = expi(math.log(x))
    return li_x / x

# === Residual δ(x) ===
def improved_delta(x, t_list, A_list, theta_list):
    return li_over_x(x) - rho(x, t_list, A_list, theta_list)

# === Lexicon Activation Score ===
def lexicon_score(A, d):
    return A / d if d > 0 else float('inf')

# === Adaptive threshold ===
def adaptive_threshold(delta_val, base=0.5, scale=0.2):
    return base * (1 + scale * abs(delta_val))

# === Structure Collapse Check ===
def verify_collapse(x_vals, t_list, A_list, theta_list, epsilon=1e-6):
    for x in x_vals:
        delta_val = improved_delta(x, t_list, A_list, theta_list)
        if abs(delta_val) > epsilon:
            print(f"x={x:.2f}, δ(x)={delta_val:.6f} > {epsilon:.6f}")
            return False
    print("All δ(x) < ε, collapse confirmed")
    return True

# === Path Expansion with adaptive ε(x) ===
def verify_adaptive_expansion(x0, t_list, A_list, theta_list, base_eps=0.5, steps=5):
    x = x0
    for step in range(steps):
        delta_val = abs(improved_delta(x, t_list, A_list, theta_list))
        eps_x = adaptive_threshold(delta_val, base=base_eps, scale=0.2)
        print(f"Step {step+1}, x={x:.2f}, δ(x)={delta_val:.6f}, ε(x)={eps_x:.6f}")
        if delta_val >= eps_x:
            print(f"Path blocked: δ(x)={delta_val:.6f} >= ε(x)={eps_x:.6f}")
            return False
        x += 0.5
    print("Path expansion successful")
    return True

# === Lexicon Activation with adaptive η_lex ===
def evaluate_adaptive_lexicon(x_vals, t_list, A_list, theta_list, base_eta=3.0, scale=0.5):
    lexicon_by_x = {}
    for x in x_vals:
        active_modes = []
        for i in range(len(t_list)):
            d = abs(li_over_x(x) - (1.0 / math.log(x) + A_list[i] * math.cos(t_list[i] * math.log(x) + theta_list[i])))
            eta_x = base_eta * (1 + scale * d)
            score = lexicon_score(A_list[i], d)
            print(f"x={x:.2f}, t={t_list[i]:.4f}, δ_n(x)={d:.6f}, score={score:.2f}, η_x={eta_x:.2f}")
            if score > eta_x:
                active_modes.append((round(t_list[i], 4), round(score, 2)))
        lexicon_by_x[round(x, 2)] = active_modes
    return lexicon_by_x

# === Run Demonstration ===
if __name__ == "__main__":
    # First 30 nontrivial zeta zeros
    t_list = [14.1347, 21.0220, 25.0108, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 
              48.0052, 49.7738, 52.9703, 56.4462, 59.3470, 60.8318, 65.1125, 
              67.0796, 69.5464, 72.0672, 75.7047, 77.1448, 79.3372, 82.9104, 
              84.7359, 87.0992, 88.8091, 92.4919, 94.6519, 95.8706, 98.8312, 101.3179]
    # Optimized amplitudes: decay with sqrt(t_i)
    A_list = [0.2 / math.sqrt(1 + t / 100) for t in t_list]  # e.g., 0.177 for t=14.1347
    # Dynamic phase shifts
    theta_list = [0.1 * i * math.sin(math.log(100 + i)) for i in range(len(t_list))]
    x_range = [100 + i for i in range(50)]

    print("▶ Structure Collapse Check (δ(x) = 0 globally?)")
    collapsed = verify_collapse(x_range, t_list, A_list, theta_list)
    print("Result:", "✅ Collapsed" if collapsed else "❌ Structure Survives")

    print("\n▶ DAG Path Expansion Check (adaptive ε(x))")
    expanded = verify_adaptive_expansion(100, t_list, A_list, theta_list)
    print("Result:", "✅ Expansion Path Exists" if expanded else "❌ Path Blocked")

    print("\n▶ Lexicon Activation Analysis (adaptive η_lex)")
    adaptive_lexicon = evaluate_adaptive_lexicon(x_range, t_list, A_list, theta_list)
    for x, modes in adaptive_lexicon.items():
        if modes:
            print(f"x = {x} → Active ψ: {modes}")
