import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
print('using skfuzzy')
confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence') #predykcja głównego modelu
similarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'similarity') #podobieństwo do referencyjnych przypadków (embeddingi)
trust = ctrl.Consequent(np.arange(0, 101, 1), 'trust')

confidence['low'] = fuzz.trimf(confidence.universe, [0.0, 0.0, 0.5])
confidence['medium'] = fuzz.trimf(confidence.universe, [0.3, 0.5, 0.7])
confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1.0, 1.0])

similarity['low'] = fuzz.trimf(similarity.universe, [0.0, 0.0, 0.5])
similarity['medium'] = fuzz.trimf(similarity.universe, [0.3, 0.5, 0.7])
similarity['high'] = fuzz.trimf(similarity.universe, [0.5, 1.0, 1.0])

trust['low'] = fuzz.trimf(trust.universe, [0, 0, 30])
trust['medium'] = fuzz.trimf(trust.universe, [20, 50, 80])
trust['high'] = fuzz.trimf(trust.universe, [70, 100, 100])

rule1 = ctrl.Rule(confidence['low'] & similarity['low'], trust['low'])
rule2 = ctrl.Rule(confidence['medium'] & similarity['medium'], trust['medium'])
rule3 = ctrl.Rule(confidence['high'] & similarity['high'], trust['high'])
rule4 = ctrl.Rule(confidence['low'] & similarity['high'], trust['medium'])
rule5 = ctrl.Rule(confidence['high'] & similarity['low'], trust['medium'])

trust_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
trust_simulator = ctrl.ControlSystemSimulation(trust_ctrl)

def evaluate_fuzzy_trust(conf, sim):
    trust_simulator.input['confidence'] = conf
    trust_simulator.input['similarity'] = sim
    trust_simulator.compute()
    return trust_simulator.output['trust']