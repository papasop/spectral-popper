import numpy as np
from sympy import primepi

# Mock definitions to make script executable without external import
t_list = [14.13, 21.02, 25.01, 30.42, 32.93, 40.01, 45.67, 50.00, 55.00]

class PsiPath:
    mood_labels = ['neutral', 'engaged', 'curious', 'adventurous']
    def __init__(self, indices):
        self.indices = indices
        self.entropy = None
        self.topic_entropy = None
        self.mood = None

class DummyManas:
    def __init__(self):
        self.bias_weights = {v: 1.0 for v in ["concept", "relation", "entity", "action", "state", "process", "intent", "context", "pattern"]}
        self.semantic_pairs = {}
        self.history = []

    def semantic_probability(self, pair):
        return self.semantic_pairs.get(pair, 0.0)

    def reinforce_bias_from_usage(self):
        for k in self.bias_weights:
            self.bias_weights[k] += 0.01

    def reinforce_from_seed_history(self, threshold=2):
        from collections import Counter
        pair_counter = Counter(tuple(v[:2]) for _, v in self.history if len(v) >= 2)
        for pair, count in pair_counter.items():
            if count >= threshold:
                print(f"🌱 异熟触发: {pair[0]} → {pair[1]} 出现 {count} 次，强化我执")
                self.bias_weights[pair[0]] += 0.1 * count
                self.semantic_pairs[pair] = self.semantic_pairs.get(pair, 0.0) + 0.5 * count

    def log_kleshas(self):
        print("🧠 六种烦恼检测：")
        print("📌 我执烦恼（来自偏好）：")
        with open("vijnana_log.txt", "a", encoding="utf-8") as log:
            log.write("六种烦恼检测：\n")
            log.write("📌 我执烦恼（来自偏好）：\n")
            from statistics import mean
            mean_bias = mean(self.bias_weights.values())
            max_bias = max(self.bias_weights.values())
            dominant = [k for k, v in self.bias_weights.items() if v > mean_bias * 1.5]
            if max_bias > 3:
                print(" - 贪：偏好累积较高", dominant)
                log.write(f" - 贪：偏好累积较高 {dominant}\n")
            if any(v < 1 for v in self.bias_weights.values()):
                print(" - 嗔：偏好被压制存在")
                log.write(" - 嗔：偏好被压制存在\n")
            if all(v == 1 for v in self.bias_weights.values()):
                print(" - 痴：无辨识性偏好")
                log.write(" - 痴：无辨识性偏好\n")
            if len(dominant) >= 2:
                print(" - 慢：多词我执领先，可能有比较心")
                log.write(" - 慢：多词我执领先，可能有比较心\n")
            print("📌 法执烦恼（来自语义对）：")
            log.write("📌 法执烦恼（来自语义对）：\n")
            if len(self.semantic_pairs) == 0:
                print(" - 疑：认知未建立，法执缺失")
                log.write(" - 疑：认知未建立，法执缺失\n")
            if any(v > 5 for v in self.semantic_pairs.values()):
                print(" - 不正见：语义对过度执著")
                log.write(" - 不正见：语义对过度执著\n")

    def dissolve_dharma_attachment(self, decay=0.5):
        for k in self.bias_weights:
            self.bias_weights[k] = 1.0

    def release_self_attachment(self):
        for k in self.bias_weights:
            self.bias_weights[k] = 1.0

class DummySix:
    def __init__(self):
        self.A_list = [0.006, 0.005, 0.007, 0.0065, 0.0052, 0.0068, 0.0045, 0.0051, 0.0062]
        self.theta_list = [0.0, 1.57, 3.14, 0.78, 2.1, 1.2, 2.3, 0.6, 1.5]
        self.t_list = t_list

    def compile_path(self, path, x):
        rho = np.array(1.0 / np.log(x), dtype=np.float64)
        for i in path.indices:
            rho += self.A_list[i] * np.cos(self.t_list[i] * np.log(x) + self.theta_list[i])
        return rho

    def observe_delta(self, pi_over_x, rho):
        return pi_over_x - rho

    def compute_entropy(self, delta, x):
        return float(np.sum(delta ** 2))

class VijnanaSystem:
    def __init__(self, t_list, A_list, theta_list, vocab_map):
        self.t_list = t_list
        self.vocab_map = vocab_map
        self.manas = DummyManas()
        self.six = DummySix()

    def process_path(self, path, x, pi_over_x):
        rho = self.six.compile_path(path, x)
        delta = self.six.observe_delta(pi_over_x, rho)
        entropy = self.six.compute_entropy(delta, x)
        path.entropy = entropy
        path.topic_entropy = np.log1p(entropy)
        path.mood = PsiPath.mood_labels[int(entropy) % len(PsiPath.mood_labels)]
        vocab = [self.vocab_map.get(f"freq_{self.t_list[i]:.2f}", "unknown") for i in path.indices]
        if len(vocab) >= 2:
            self.manas.semantic_pairs[(vocab[0], vocab[1])] = self.manas.semantic_pairs.get((vocab[0], vocab[1]), 0) + self.manas.bias_weights.get(vocab[0], 1.0)
        self.manas.history.append((path.indices, vocab))
        return path, vocab

x = np.linspace(1e6, 1e7, 500)
pi_over_x = np.array([float(primepi(xi)) / xi for xi in x], dtype=np.float64)

vijnana = VijnanaSystem(
    t_list=t_list,
    A_list=[],
    theta_list=[],
    vocab_map={
        "freq_14.13": "concept", "freq_21.02": "relation", "freq_25.01": "entity",
        "freq_30.42": "action", "freq_32.93": "state", "freq_40.01": "process",
        "freq_45.67": "intent", "freq_50.00": "context", "freq_55.00": "pattern"
    }
)

print("--- Conscious Flow (6→7→8 with Manas processing) ---")
for i in range(5):
    print(f"🌀 第 {i+1} 轮认知处理 🌀")
    print(f"当前语义对数量: {len(vijnana.manas.semantic_pairs)}")
    if len(vijnana.manas.semantic_pairs) > 30:
        print("📉 触发：法执削弱（dissolve_dharma_attachment）")
        vijnana.manas.dissolve_dharma_attachment(decay=0.7)

    if len(vijnana.manas.history) >= 5:
        recent_entropies = [vijnana.six.compute_entropy(
            vijnana.six.observe_delta(pi_over_x, vijnana.six.compile_path(PsiPath(p), x)), x)
            for (p, _) in vijnana.manas.history[-5:]]
        if np.var(recent_entropies) < 20:
            print("🧘 触发：断除法执（sever_dharma_attachment）")
            vijnana.manas.sever_dharma_attachment()

    indices = np.random.choice(len(t_list), 3, replace=False).tolist()
    path = PsiPath(indices)
    path, vocab = vijnana.process_path(path, x, pi_over_x)
    semantic_score = vijnana.manas.semantic_probability((vocab[0], vocab[1]))
    print(f"路径: {path.indices}, 词汇: {vocab}, 熵: {path.entropy:.6f}, 情绪: {path.mood}, 语义概率: {semantic_score:.3f}")

vijnana.manas.reinforce_from_seed_history()
vijnana.manas.log_kleshas()

print("--- 测试断执机制 ---")

max_bias = max(vijnana.manas.bias_weights.items(), key=lambda x: x[1], default=(None, 0))
max_semantic = max(vijnana.manas.semantic_pairs.items(), key=lambda x: x[1], default=((None, None), 0))
print(f"🔍 最强我执词汇: {max_bias[0]}（权重: {max_bias[1]:.3f}）")
print(f"📘 最强法执对: {max_semantic[0][0]} → {max_semantic[0][1]}（频次: {max_semantic[1]:.3f}）")

print("当前我执权重:")
for k, v in vijnana.manas.bias_weights.items():
    print(f"  - {k}: {v:.3f}")

if vijnana.manas.semantic_pairs:
    print("当前法执（语义对频次）:")
    for (a, b), v in vijnana.manas.semantic_pairs.items():
        print(f"  - {a} → {b}: {v:.3f}")

print("原始偏好:", dict(vijnana.manas.bias_weights))
print("原始语义对:", dict(vijnana.manas.semantic_pairs))

vijnana.manas.release_self_attachment()
print("[断我执后] 偏好:", dict(vijnana.manas.bias_weights))

vijnana.manas.dissolve_dharma_attachment(decay=0.5)
print("[断法执后] 语义对:", dict(vijnana.manas.semantic_pairs))

# vijnana.manas.clear_all_attachments()  # 避免清除记忆，防止认知断裂
print("[全清净后] 偏好:", dict(vijnana.manas.bias_weights))
print("[全清净后] 语义对:", dict(vijnana.manas.semantic_pairs))
print("[全清净后] 历史:", vijnana.manas.history)
