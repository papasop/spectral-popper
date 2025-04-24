# Mini-psiDial: Psi Dialogue System with Goal-Driven Psi Agents and Multi-Agent Co-Resonance
import random
import matplotlib.pyplot as plt

# --- Lexicon: psi paths with their residuals ---
lexicon = {
    'psi1': {'next': [('psi2', 0.1), ('psi3', 0.3)]},
    'psi3': {'next': [('psi4', 0.3)]},
    'psi2': {'next': []},
    'psi4': {'next': []}
}

# --- DAG: Allowed transitions ---
dag = {
    'psi1': ['psi2', 'psi3'],
    'psi3': ['psi4'],
    'psi2': [],
    'psi4': []
}

# Psi Agent with goals, memory, emotion, and adaptive behavior
class PsiAgent:
    def __init__(self, name, epsilon=0.2, mood='neutral', goal='max_leaps'):
        self.name = name
        self.epsilon = epsilon
        self.mood = mood
        self.memory = []
        self.collapse_count = 0
        self.goal = goal

    def decide(self, current, injected_options=None):
        options = lexicon.get(current, {}).get('next', [])
        if injected_options:
            options += injected_options
        if not options:
            return None

        # Adjust epsilon by mood
        mood_bias = {'neutral': 0.0, 'adventurous': 0.2, 'conservative': -0.1}
        effective_epsilon = max(0.0, min(1.0, self.epsilon + mood_bias.get(self.mood, 0.0)))

        # Forget old paths
        if random.random() < 0.3 and self.memory:
            self.memory.pop(0)

        new_options = [opt for opt in options if opt[0] not in self.memory]
        if new_options:
            options = new_options

        # Goal-driven prioritization
        if self.goal == 'max_delta':
            sorted_opts = sorted(options, key=lambda x: -x[1])  # high δ
        elif self.goal == 'min_collapse':
            sorted_opts = sorted(options, key=lambda x: abs(x[1] - 0.2))  # avoid collapse zone
        else:  # default to 'max_leaps'
            sorted_opts = sorted(options, key=lambda x: x[1])

        choice = sorted_opts[-1] if random.random() < effective_epsilon else sorted_opts[0]
        self.memory.append(choice[0])
        return choice

    def adapt_mood(self):
        if self.collapse_count >= 2:
            self.mood = 'adventurous'
        elif self.collapse_count == 1:
            self.mood = 'neutral'
        else:
            self.mood = 'conservative'

# PsiB injects perturbations
class PsiInjector:
    def inject(self, current):
        if current == 'psi1':
            return [('psi3', 0.3)]
        return []

# Collapse detector
class CollapseDetector:
    def check_collapse(self, path_with_deltas, threshold=0.15):
        low_residuals = [delta for _, delta in path_with_deltas if delta < threshold]
        return len(low_residuals) >= len(path_with_deltas) // 2

# Run interaction between two agents (co-resonance)
def run_multi_agent_dialogue(agentA, agentB, injector, detector, start='psi1'):
    path = [start]
    path_with_deltas = []
    current = start
    agentA.memory = [start]
    agentB.memory = [start]
    turn = 0

    while True:
        injected = injector.inject(current)
        active_agent = agentA if turn % 2 == 0 else agentB
        next_step = active_agent.decide(current, injected_options=injected)
        if not next_step:
            break
        next_node, delta = next_step
        path.append(next_node)
        path_with_deltas.append((next_node, delta))
        current = next_node
        turn += 1

    collapsed = detector.check_collapse(path_with_deltas)
    if collapsed:
        agentA.collapse_count += 1
        agentB.collapse_count += 1
    agentA.adapt_mood()
    agentB.adapt_mood()
    return path, path_with_deltas, collapsed, agentA.mood, agentB.mood

# Simulation
def simulate_multi_runs(n=5):
    random.seed(42)
    injector = PsiInjector()
    detector = CollapseDetector()

    all_deltas = []
    for i in range(n):
        agentA = PsiAgent("ψA", epsilon=0.3, goal='max_delta')
        agentB = PsiAgent("ψB", epsilon=0.2, goal='min_collapse')
        path, deltas, collapsed, moodA, moodB = run_multi_agent_dialogue(agentA, agentB, injector, detector)
        print(f"Run {i+1}: Path = {path} | Collapsed = {collapsed} | MoodA = {moodA}, MoodB = {moodB}")
        all_deltas.append([d for _, d in deltas])

    plt.figure(figsize=(10, 5))
    for i, deltas in enumerate(all_deltas):
        plt.plot(deltas, label=f"Run {i+1}")
    plt.xlabel("Step")
    plt.ylabel("Residual δ")
    plt.title("Multi-Agent ψ Dialogue: Goal-Driven Co-Resonance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

simulate_multi_runs()
