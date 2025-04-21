import numpy as np
import math
import hashlib
from scipy.optimize import minimize

# ---- 参数配置 ----
x = 100000                             # 输入自然数 x
true_density = 9592 / x               # π(x)/x = 0.09592
target_error = 1e-6                   # 精度目标
lambda_reg = 0.0001                   # 正则项
zeta_zeros = [                        # 前 15 个 ζ 零点
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446247, 59.347044, 60.831779, 65.112544
]

# ---- 共振函数定义 ----
logx = math.log(x)
used_freqs = []

while True:
    remaining = [t for t in zeta_zeros if t not in used_freqs]
    if not remaining:
        break
    next_t = sorted(remaining, key=lambda t: abs((t * logx) % (2 * np.pi) - np.pi))[0]
    used_freqs.append(next_t)
    N = len(used_freqs)
    init_A = np.ones(N)
    init_theta = np.zeros(N)
    init_params = np.concatenate([init_A, init_theta])
    bounds = [(0.01, 2)] * N + [(-np.pi, np.pi)] * N

    def rho_ads(params):
        s = 1 / logx
        for i in range(N):
            s += params[i] * np.cos(used_freqs[i] * logx + params[N + i])
        return s

    def loss(params):
        return (rho_ads(params) - true_density)**2 + lambda_reg * np.sum(np.exp(-params[:N]))

    result = minimize(loss, init_params, bounds=bounds, method="L-BFGS-B")
    error = abs(rho_ads(result.x) - true_density)
    if error <= target_error:
        break

# ---- Merkle 哈希函数 ----
def merkle_signature(freqs, A_vals, theta_vals):
    data = ''.join(f"{t:.6f}{A:.6f}{theta:.6f}" for t, A, theta in zip(freqs, A_vals, theta_vals))
    return hashlib.sha256(data.encode()).hexdigest()

signature_original = merkle_signature(used_freqs, result.x[:N], result.x[N:])
# 改动 θ₁ 测试签名是否变化
theta_perturbed = result.x[N:].copy()
theta_perturbed[0] += 0.01
signature_modified = merkle_signature(used_freqs, result.x[:N], theta_perturbed)

# ---- 输出结构验证结果 ----
print("\n📌 结构路径验证结果")
print(f"输入 x = {x}, log(x) ≈ {logx:.4f}")
print(f"真实密度 π(x)/x = {true_density:.10f}")
print(f"AdS拟合密度       = {rho_ads(result.x):.10f}")
print(f"拟合误差           = {error:.2e}")
print(f"结构激活能量 ∑A    = {np.sum(result.x[:N]):.4f}")
print(f"使用 ζ 零点频率数  = {N}")
print("频率组合 t_n:")
for i, t in enumerate(used_freqs):
    print(f"  t_{i+1} = {t:.6f}")

print("\n🔧 幅度 A:")
print(np.round(result.x[:N], 6))
print("\n🔧 相位 θ:")
print(np.round(result.x[N:], 6))

print("\n🔐 Merkle 路径签名:")
print("原始路径签名      =", signature_original)
print("扰动后路径签名    =", signature_modified)
print("签名是否一致？     =", "✅ 是" if signature_original == signature_modified else "❌ 否")
