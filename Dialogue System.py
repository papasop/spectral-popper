# StructureLang Semantic Mapping + Multi-Agent Resonance Expansion (ψA + ψB → ψC → ψD) with Explicit DAG, Agent Goals, and Dynamic Emotion
import random
import numpy as np

# Frequency Lexicon Snapshot (24.5)
frequency_lexicon = {
    'psi1': {'t': 14.13, 'A': 0.0083, 'theta': 0.0},
    'psi2': {'t': 21.02, 'A': 0.0041, 'theta': 1.57},
    'psi3': {'t': 25.01, 'A': 0.0062, 'theta': 3.14},
    'psi4': {'t': 30.42, 'A': 0.0054, 'theta': 0.78},
    'psi5': {'t': 32.93, 'A': 0.0029, 'theta': 2.10},
    'psi6': {'t': 35.00, 'A': 0.0048, 'theta': 1.0},
    'psi7': {'t': 35.12, 'A': 0.0035, 'theta': 2.5},
    'psi8': {'t': 41.00, 'A': 0.0036, 'theta': 0.9},
    'psi9': {'t': 43.50, 'A': 0.0040, 'theta': 1.2},
    'psi10': {'t': 45.70, 'A': 0.0027, 'theta': 2.1},
    'psi11': {'t': 48.12, 'A': 0.0032, 'theta': 2.8},
    'psi12': {'t': 50.33, 'A': 0.0029, 'theta': 1.4}
}

# DAG with extended nodes
DAG = {
    'psi1': ['psi2', 'psi3'],
    'psi2': ['psi5', 'psi6', 'psi8'],
    'psi3': ['psi4', 'psi9'],
    'psi4': ['psi5', 'psi6'],
    'psi5': ['psi7', 'psi8'],
    'psi6': ['psi7', 'psi9'],
    'psi7': ['psi10'],
    'psi8': ['psi10'],
    'psi9': ['psi10'],
    'psi10': ['psi11'],
    'psi11': ['psi12'],
    'psi12': []
}

def compute_tscore(deltas):
    accuracy = 1 - np.mean(deltas)
    tension = np.mean(deltas)
    activation = max(deltas)
    continuity = 1.0
    composability = 1.0
    return 0.2 * accuracy + 0.2 * tension + 0.2 * activation + 0.2 * continuity + 0.2 * composability

def compute_delta_path(path):
    return [frequency_lexicon[n]['A'] / frequency_lexicon[n]['t'] for n in path]

def generate_paths(start, path=None, max_depth=10):
    if path is None:
        path = [start]
    current = path[-1]
    if not DAG[current] or len(path) >= max_depth:
        return [path]
    paths = []
    for nxt in DAG[current]:
        paths.extend(generate_paths(nxt, path + [nxt], max_depth))
    return paths

def compute_spsi(deltas):
    return -1 * np.sum([d * np.log(d + 1e-9) for d in deltas])

def phi_trigger(delta, threshold=0.00015):
    return max(delta) > threshold

def delta_overlap(delta_a, delta_b, tol=1e-5):
    return any(abs(a - b) < tol for a in delta_a for b in delta_b)

class Agent:
    def __init__(self, name, goal):
        self.name = name
        self.goal = goal
        self.mood = 'neutral'

    def evaluate(self, deltas):
        avg = np.mean(deltas)
        if avg > 0.00025:
            self.mood = 'engaged'
        elif avg < 0.0001:
            self.mood = 'collapsing'
        else:
            self.mood = 'neutral'
        return avg

if __name__ == '__main__':
    # Initialize agents
    agentA = Agent("ψA", goal='delta')
    agentB = Agent("ψB", goal='tscore')

    # Generate all paths and select 10 (5 for ψA, 5 for ψB)
    all_paths = generate_paths('psi1', max_depth=10)
    random.shuffle(all_paths)  # Randomize to ensure variety
    total_paths = min(10, len(all_paths))  # Ensure up to 10 paths
    selected_paths = all_paths[:total_paths]
    ψA_paths = selected_paths[:total_paths//2]  # First 5 for ψA
    ψB_paths = selected_paths[total_paths//2:]  # Last 5 for ψB

    print(f"Generated {total_paths} DAG paths from ψ1 (ψA: {len(ψA_paths)}, ψB: {len(ψB_paths)})")

    # Track consensus and escapes
    consensus_count = 0
    escape_count = 0
    path_pairs = []

    for a_path, b_path in zip(ψA_paths, ψB_paths):
        # Compute deltas for both agents
        deltas_a = compute_delta_path(a_path)
        deltas_b = compute_delta_path(b_path)

        # Agent evaluation
        avg_delta_a = agentA.evaluate(deltas_a)
        avg_delta_b = agentB.evaluate(deltas_b)

        # Compute metrics
        overlap = delta_overlap(deltas_a, deltas_b)
        consensus = overlap and 'psi12' in a_path and 'psi12' in b_path and abs(avg_delta_a - avg_delta_b) < 0.00005
        spsi_a = compute_spsi(deltas_a)
        phi_a = phi_trigger(deltas_a)
        tscore_a = compute_tscore(deltas_a)

        # Adjusted classification logic for diversity
        if spsi_a > 0.01 and consensus:
            classification = "emergent structure"
        elif avg_delta_a < 0.0001 or not overlap:
            classification = "escaped path"
        else:
            classification = "moderate"

        if consensus:
            consensus_count += 1
        else:
            escape_count += 1

        path_pairs.append({
            'ψA_path': a_path,
            'ψB_path': b_path,
            'avg_delta_a': avg_delta_a,
            'spsi_a': spsi_a,
            'tscore_a': tscore_a,
            'phi_a': phi_a,
            'classification': classification,
            'overlap': overlap,
            'mood_a': agentA.mood,
            'mood_b': agentB.mood,
            'consensus': consensus
        })

    # Output results
    for pair in path_pairs:
        a_path = pair['ψA_path']
        print(f"{agentA.name} → {' → '.join(a_path)} | avg δ = {pair['avg_delta_a']:.6f} | "
              f"Sψ = {pair['spsi_a']:.6f} | t_score = {pair['tscore_a']:.4f} | φ-trigger = {pair['phi_a']} | "
              f"type = {pair['classification']} | overlap = {pair['overlap']} | mood = {pair['mood_a']}")
        b_path = pair['ψB_path']
        deltas_b = compute_delta_path(b_path)
        spsi_b = compute_spsi(deltas_b)
        tscore_b = compute_tscore(deltas_b)
        phi_b = phi_trigger(deltas_b)
        print(f"{agentB.name} → {' → '.join(b_path)} | avg δ = {np.mean(deltas_b):.6f} | "
              f"Sψ = {spsi_b:.6f} | t_score = {tscore_b:.4f} | φ-trigger = {phi_b} | "
              f"type = {pair['classification']} | overlap = {pair['overlap']} | mood = {pair['mood_b']}")
        print(f"Consensus: {pair['consensus']}\n")

    # Compute residual thresholds
    threshold_high = 0.005
    threshold_low = 0.001
    above_thresh = 0
    below_thresh = 0

    for path in selected_paths:
        deltas = compute_delta_path(path)
        avg_delta = np.mean(deltas)
        if avg_delta > threshold_high:
            above_thresh += 1
        elif avg_delta <= threshold_low:
            below_thresh += 1

    print(f"Total Consensus: {consensus_count}, Escapes: {escape_count}, Total Paths Compared: {len(path_pairs)}")
    print(f"Paths avg δ > {threshold_high}: {above_thresh}, avg δ <= {threshold_low}: {below_thresh}")
    print(f"δ-threshold: {threshold_low}")
