import math
import numpy as np
from scipy.special import expi
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === Nontrivial Zeta Zeros (Extended to 150) ===
t_list = [
    14.1347, 21.0220, 25.0108, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 48.0052, 49.7738,
    52.9703, 56.4462, 59.3470, 60.8318, 65.1125, 67.0796, 69.5464, 72.0672, 75.7047, 77.1448,
    79.3372, 82.9104, 84.7359, 87.0992, 88.8091, 92.4919, 94.6519, 95.8706, 98.8312, 101.3179,
    103.7255, 105.4465, 107.1684, 111.0295, 111.8746, 114.3202, 116.2287, 118.7905, 121.3702, 122.9468,
    124.2569, 127.5165, 129.5786, 131.0871, 133.8373, 135.5065, 137.2946, 139.7350, 141.1235, 143.1116,
    146.0009, 147.4226, 150.0536, 150.9251, 153.0247, 156.1127, 157.5974, 158.8499, 161.1887, 163.0305,
    165.5370, 167.1844, 169.0946, 169.9118, 172.7605, 174.7547, 176.4414, 178.3772, 180.2448, 181.6848,
    184.8745, 186.5920, 187.2288, 189.4166, 191.2842, 193.0795, 195.2653, 196.8761, 198.0157, 201.2647,
    202.4930, 204.1895, 206.1648, 207.0727, 209.5766, 211.6902, 213.3473, 214.5470, 216.1690, 218.3155,
    220.7145, 221.4303, 224.0070, 224.9833, 227.4211, 229.3370, 230.3320, 231.9871, 234.2146, 236.5243,
    238.8192, 240.7447, 243.0616, 245.3892, 246.7084, 248.9334, 251.3098, 252.8717, 255.2587, 257.1499,
    259.5557, 261.2231, 263.4439, 265.3187, 267.2239, 269.7198, 271.1358, 273.3298, 275.3129, 277.0797,
    279.2290, 281.3466, 283.0898, 285.2477, 287.1183, 289.2227, 291.2698, 293.0232, 295.1327, 297.2287,
    299.0215, 301.0194, 303.0487, 305.1478, 306.8557, 308.8977, 310.8794, 312.5466, 314.6850, 316.8269
]

# === ψ_n(x) Definition ===
def psi(t, x, A=1.0, theta=0.0):
    return A * math.cos(t * math.log(x) + theta) if x > 1 else 0.0

# === Structure Density ρ(x) ===
def rho(x, t_list, A_list, theta_list):
    base = 1.0 / math.log(x) if x > 1 else 0.0
    modal_sum = sum(A_list[i] * psi(t_list[i], x, 1.0, theta_list[i]) for i in range(len(t_list)))
    return base + modal_sum

# === π(x)/x Approximation using li(x)/x ===
def li_over_x(x):
    if x < 2:
        return 0.0
    return expi(math.log(x)) / x

# === Residual δ(x) ===
def delta(x, t_list, A_list, theta_list):
    return li_over_x(x) - rho(x, t_list, A_list, theta_list)

# === Residual Derivative ===
def residual_derivative(x, t_list, A_list, theta_list, h=1e-6):
    delta_plus = delta(x + h, t_list, A_list, theta_list)
    delta_minus = delta(x - h, t_list, A_list, theta_list)
    return (delta_plus - delta_minus) / (2 * h)

# === Lexicon Activation Score ===
def lexicon_score(A, d):
    return A / d if d > 0 else float('inf')

# === Optimized Adaptive Thresholds ===
def adaptive_epsilon(delta_val, base=0.1, scale=0.1):
    return base * (1 + scale * abs(delta_val))

def adaptive_eta(delta_val, base=1.0, scale=0.2):
    return base * (1 + scale * abs(delta_val))

# === Optimize Amplitudes and Phases ===
def optimize_params(x_vals, t_list, initial_A, initial_theta):
    def objective(params):
        A_list = params[:len(t_list)]
        theta_list = params[len(t_list):]
        total_delta = sum(abs(delta(x, t_list, A_list, theta_list))**2 for x in x_vals)
        return total_delta
    initial_guess = np.concatenate([initial_A, initial_theta])
    bounds = [(0, 1)] * len(t_list) + [(-np.pi, np.pi)] * len(t_list)
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
    A_list = result.x[:len(t_list)].tolist()
    theta_list = result.x[len(t_list):].tolist()
    return A_list, theta_list

# === Structure Collapse Check ===
def verify_collapse(x_vals, t_list, A_list, theta_list, epsilon=1e-4):
    print("\n=== Testing Collapse (δ(x) ≈ 0 globally) ===")
    deltas = [delta(x, t_list, A_list, theta_list) for x in x_vals]
    results = [abs(d) < epsilon for d in deltas]
    for x, d, r in zip(x_vals, deltas, results):
        print(f"x={x:.2f}, δ(x)={d:.6f}, |δ(x)| < {epsilon:.6f}: {r}")
    if all(results):
        print("All δ(x) < ε, collapse confirmed")
        return True
    print("Nonzero δ(x) detected, structure survives")
    return False

# === Zero Residual Test ===
def test_zero_residual(x_vals, t_list, epsilon=1e-4):
    print("\n=== Testing Zero Residual Case ===")
    initial_A = [0.01] * len(t_list)
    initial_theta = [0.0] * len(t_list)
    A_list, theta_list = optimize_params(x_vals, t_list, initial_A, initial_theta)
    print("Optimized Amplitudes (first 5):", [round(a, 6) for a in A_list[:5]])
    print("Optimized Phases (first 5):", [round(t, 6) for t in theta_list[:5]])
    deltas = [delta(x, t_list, A_list, theta_list) for x in x_vals]
    results = [abs(d) < epsilon for d in deltas]
    for x, d, r in zip(x_vals, deltas, results):
        print(f"x={x:.2f}, δ(x)={d:.6f}, |δ(x)| < {epsilon:.6f}: {r}")
    if all(results):
        print("Zero residual achieved, testing generativity...")
        lexicon = evaluate_lexicon([x_vals[0]], t_list[:20], A_list[:20], theta_list[:20])
        expanded, path_lengths = verify_adaptive_expansion(x_vals[0], t_list, A_list, theta_list, steps=5)
        if not lexicon[x_vals[0]] and not expanded:
            print("✅ Empty lexicon and no path expansion, collapse confirmed")
            return True
        else:
            print(f"❌ Generativity detected: Lexicon={lexicon[x_vals[0]]}, Expanded={expanded}")
            return False
    print("Nonzero residuals, zero residual test inconclusive")
    return None

# === DAG Path Expansion ===
def verify_adaptive_expansion(x0, t_list, A_list, theta_list, base_eps=0.1, steps=10):
    print("\n=== Testing DAG Path Expansion ===")
    x = x0
    path_lengths = []
    for step in range(steps):
        delta_val = abs(delta(x, t_list, A_list, theta_list))
        eps_x = adaptive_epsilon(delta_val)
        print(f"Step {step+1}, x={x:.2f}, δ(x)={delta_val:.6f}, ε(x)={eps_x:.6f}")
        if delta_val >= eps_x:
            print(f"Path blocked: δ(x)={delta_val:.6f} >= ε(x)={eps_x:.6f}")
            return False, path_lengths
        path_lengths.append(step + 1)
        x += (1000 - x0) / steps
    print("Path expansion successful")
    return True, path_lengths

# === Lexicon Activation ===
def evaluate_lexicon(x_vals, t_list, A_list, theta_list, base_eta=1.0, scale=0.2):
    print("\n=== Lexicon Activation Analysis ===")
    lexicon_by_x = {}
    for x in x_vals:
        active_modes = []
        delta_vals = []
        for i in range(len(t_list)):
            d = abs(li_over_x(x) - (1.0 / math.log(x) + A_list[i] * psi(t_list[i], x, 1.0, theta_list[i])))
            delta_vals.append(d)
            eta_x = adaptive_eta(d, base_eta, scale)
            score = lexicon_score(A_list[i], d)
            print(f"x={x:.2f}, t={t_list[i]:.4f}, δ_n(x)={d:.6f}, score={score:.2f}, η_x={eta_x:.2f}")
            if score > eta_x:
                active_modes.append((t_list[i], round(score, 2)))
        lexicon_by_x[x] = active_modes
        if not active_modes:
            print(f"x={x:.2f}: Empty lexicon, testing generativity falsification")
        else:
            print(f"x={x:.2f}: Active modes: {active_modes}")
    return lexicon_by_x

# === Residual Statistics ===
def residual_statistics(x_vals, t_list, A_list, theta_list):
    print("\n=== Residual Statistics ===")
    deltas = [delta(x, t_list, A_list, theta_list) for x in x_vals]
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    max_delta = np.max(np.abs(deltas))
    print(f"Mean δ(x): {mean_delta:.6f}, Std δ(x): {std_delta:.6f}, Max |δ(x)|: {max_delta:.6f}")
    return mean_delta, std_delta, max_delta

# === Plot Residuals ===
def plot_residuals(x_vals, t_list, A_list, theta_list):
    deltas = [delta(x, t_list, A_list, theta_list) for x in x_vals]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, deltas, label="δ(x)")
    plt.axhline(0, color='k', linestyle='--')
    plt.title("Residual δ(x) After Optimization")
    plt.xlabel("x")
    plt.ylabel("δ(x)")
    plt.legend()
    plt.show()

# === Run Demonstration ===
if __name__ == "__main__":
    # Parameters
    A_list = [0.1 / math.sqrt(1 + t / 100) for t in t_list]
    theta_list = [0.05 * i * math.sin(math.log(100 + i)) for i in range(len(t_list))]
    x_vals = np.linspace(10, 1000, 50)

    # Residual Statistics (Before Optimization)
    residual_statistics(x_vals, t_list, A_list, theta_list)

    # Test Zero Residual Case
    zero_residual_result = test_zero_residual(x_vals, t_list, epsilon=1e-4)
    print("Zero Residual Test:", "✅ Collapse Confirmed" if zero_residual_result else "❌ Falsified or Inconclusive")

    # Plot Residuals
    plot_residuals(x_vals, t_list, A_list, theta_list)
