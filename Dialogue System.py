# StructureLang v2.7: Deep Agent Interaction, Long-Term Entropy, and Goal Conflict Simulation
import random
import numpy as np
import matplotlib.pyplot as plt

class CollapseDetector:
    def __init__(self, base_epsilon=0.15):
        self.epsilon = base_epsilon
        self.entropy_log = []

    def check_collapse(self, residuals):
        low_count = len([r for r in residuals if r < self.epsilon])
        return low_count / len(residuals) >= 0.5

    def adjust_epsilon(self, recent_residuals):
        avg = np.mean(recent_residuals)
        std = np.std(recent_residuals)
        entropy = -1 * np.sum([r * np.log(r + 1e-9) for r in recent_residuals])
        self.entropy_log.append(entropy)

        if avg < 0.15:
            self.epsilon = max(0.05, self.epsilon - 0.01)
        elif std > 0.1:
            self.epsilon = min(0.3, self.epsilon + 0.01)
        return self.epsilon, entropy

class PsiInjector:
    def inject(self, current):
        return [('psi3', 0.3), ('psi4', 0.25)] if current == 'psi1' else []

class PsiLogger:
    def __init__(self):
        self.logs = []
        self.moods = []
        self.tscores = []

    def log(self, agent_name, step, choice, delta, mood, tscore, reason=""):
        msg = f"[step {step}] {agent_name} chose {choice}. δ={delta:.2f}, tscore={tscore:.2f}, mood={mood}, reason: {reason}"
        self.logs.append(msg)
        self.moods.append(mood)
        self.tscores.append(tscore)
        print(msg)

    def plot_tscore_vs_mood(self):
        mood_numeric = [0 if m == 'conservative' else 1 if m == 'neutral' else 2 for m in self.moods]
        plt.figure(figsize=(8, 4))
        plt.plot(self.tscores, label='t_score')
        plt.plot(mood_numeric, label='Mood (0: consv, 1: neutral, 2: advnt)')
        plt.xlabel('Step')
        plt.title('t_score vs Mood Evolution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class DAG:
    def __init__(self):
        self.nodes = {
            'psi1': [('psi2', 0.1), ('psi3', 0.3)],
            'psi2': [('psi5', 0.15), ('psi6', 0.22)],
            'psi3': [('psi4', 0.25)],
            'psi4': [('psi5', 0.2), ('psi6', 0.28)],
            'psi5': [('psi7', 0.18)],
            'psi6': [('psi7', 0.25)],
            'psi7': []
        }

    def get_options(self, node):
        return self.nodes.get(node, [])

class GoalDrivenAgent:
    def __init__(self, name, goal='max_delta', epsilon=0.2, mood='neutral'):
        self.name = name
        self.goal = goal
        self.epsilon = epsilon
        self.mood = mood
        self.memory = []

    def decide(self, current, dag, injected_options=None):
        options = dag.get_options(current) + (injected_options or [])
        if not options:
            return None

        mood_bias = {'neutral': 0.0, 'adventurous': 0.2, 'conservative': -0.1}
        effective_epsilon = max(0.0, min(1.0, self.epsilon + mood_bias.get(self.mood, 0.0)))

        scored_options = [(opt, delta, delta * random.uniform(0.8, 1.2)) for opt, delta in options]

        if self.goal == 'max_delta':
            scored_options.sort(key=lambda x: -x[1])
        elif self.goal == 'max_tscore':
            scored_options.sort(key=lambda x: -x[2])
        else:
            scored_options.sort(key=lambda x: x[1])

        choice = scored_options[-1] if random.random() < effective_epsilon else scored_options[0]
        self.memory.append(choice[0])
        return choice[0], choice[1], choice[2]

    def adapt_mood(self, collapse_count):
        if collapse_count >= 2:
            self.mood = 'adventurous'
        elif collapse_count == 1:
            self.mood = 'neutral'
        else:
            self.mood = 'conservative'

class PsiCGenerator:
    def __init__(self):
        self.shared_nodes = []

    def update(self, a_path, b_path):
        overlap = set(a_path).intersection(set(b_path))
        self.shared_nodes = sorted(list(overlap))

    def generate_consensus(self):
        if self.shared_nodes:
            return f"ψC path: {' ⊕ '.join(self.shared_nodes)}"
        return "ψC undefined (no shared structure)"

class SemanticEntropyMonitor:
    def __init__(self):
        self.delta_series = []
        self.entropy_series = []

    def update(self, delta_segment):
        entropy = -1 * np.sum([d * np.log(d + 1e-9) for d in delta_segment])
        self.delta_series.append(np.mean(delta_segment))
        self.entropy_series.append(entropy)

    def plot_trends(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.delta_series, label='δ(x) mean')
        plt.plot(self.entropy_series, label='Sψ entropy')
        plt.title("Residual δ vs Semantic Entropy")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class RepairEngine:
    def repair(self, dag, current):
        options = dag.get_options(current)
        if not options:
            new_nodes = list(dag.nodes.keys())
            return random.choice(new_nodes), 0.2, 0.2
        return options[0][0], options[0][1], options[0][1] * 1.0

if __name__ == '__main__':
    dag = DAG()
    cd = CollapseDetector()
    injector = PsiInjector()
    logger = PsiLogger()
    entropy_monitor = SemanticEntropyMonitor()
    repair_engine = RepairEngine()
    consensus = PsiCGenerator()

    agentA = GoalDrivenAgent("ψA", goal='max_delta')
    agentB = GoalDrivenAgent("ψB", goal='max_tscore')

    pathA = ['psi1']
    pathB = ['psi1']
    delta_log = []
    collapse_count = 0

    for step in range(100):
        current = pathA[-1] if step % 2 == 0 else pathB[-1]
        injected = injector.inject(current)
        active_agent = agentA if step % 2 == 0 else agentB
        decision = active_agent.decide(current, dag, injected_options=injected)

        if not decision:
            next_node, delta, tscore = repair_engine.repair(dag, current)
        else:
            next_node, delta, tscore = decision

        if step % 2 == 0:
            pathA.append(next_node)
        else:
            pathB.append(next_node)

        delta_log.append(delta)
        collapsed = cd.check_collapse(delta_log[-5:])
        if collapsed:
            collapse_count += 1
        eps, _ = cd.adjust_epsilon(delta_log[-5:])
        active_agent.adapt_mood(collapse_count)
        logger.log(active_agent.name, step, next_node, delta, active_agent.mood, tscore, reason="goal-driven test")
        entropy_monitor.update(delta_log[-5:])

    logger.plot_tscore_vs_mood()
    entropy_monitor.plot_trends()
    consensus.update(pathA, pathB)
    print(consensus.generate_consensus())

    agentC_path = consensus.shared_nodes
    if agentC_path:
        print(f"ψC emergent consensus path (shared by ψA and ψB): {agentC_path}")
    else:
        print("No ψC consensus path formed in this run.")
