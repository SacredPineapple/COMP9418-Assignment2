from smart_building import SmartBuilding
from example_test import SmartBuildingSimulatorExample
import scipy
import numpy as np
import time

COST_PERSON_NO_LIGHT = 4
COST_LIGHT = 1

def info_to_actions(mus, vars):
    sigmas = np.sqrt(vars)
    pow = - mus**2 / (2*vars)
    term1 = ( sigmas / np.sqrt(2*np.pi) ) * np.exp(pow)
    term2 = mus * scipy.stats.norm.sf(0, loc=mus, scale=sigmas)
    cost_light_off = (term1 + term2) * COST_PERSON_NO_LIGHT

    # for mu, var, cost in zip(mus, vars, cost_light_off):
    #     print(f"N({mu}, {var}) has cost {cost})")
    
    light_on = cost_light_off > COST_LIGHT
    actions_dict = {'lights' + str(i+1): 'on' if light_on[i] else 'off' for i in range(34)}
    return actions_dict

def get_action(sensor_data, smart_building):
    smart_building.tick()
    smart_building.apply_evidence(sensor_data)
    means, vars = smart_building.query()
    actions = info_to_actions(means, vars)
    return actions

# Gets the cost associated with a transition matrix
# Cost is defined as the cents spent over the day, PLUS a regularization
# term (lambda * [number of non-zero terms in transition_matrix])
def cost_t_matrix(transition_matrix):
    total_cost = 0
    lam = 0 # TODO: Figure out a decent regularisation term (maybe look if outputs after grid-searchy thing makes sense)

    smart_building = SmartBuilding(transition_matrix)
    simulator = SmartBuildingSimulatorExample()

    for i in range(len(simulator.data)):
        sensor_data = simulator.timestep()
        actions_dict = get_action(sensor_data, smart_building)   
        total_cost += simulator.cost_timestep(actions_dict)
    return total_cost + lam * np.count_nonzero(transition_matrix)

# # Use my manually set thingoes as a head-start
# from data_store import getTransitions
# t_m = np.zeros((37, 37))
# for i, ns in getTransitions().items():
#     for j, k in ns:
#         t_m[i, j] = k

if __name__ == "__main__":
    start_time = time.perf_counter()

    n_iterations, step_size = 100, 0.1
    start_matrix = np.eye(37)
    # Dict of Start : [Possible Ends]. Handmade
    neighboursDict = {
        0: [14, 22, 24, 35],
        1: [2, 3],
        2: [1, 3, 12, 36],
        3: [1, 2, 12, 36],
        4: [5, 6],
        5: [4, 6],
        6: [4, 5, 14, 35],
        7: [8, 9, 15, 16, 36],
        8: [7, 9, 10, 15, 16, 36],
        9: [7, 8, 10, 11, 15, 16, 36],
        10: [8, 9, 11, 12, 16, 17, 36],
        11: [9, 10, 12, 13, 16, 17, 18, 35],
        12: [2, 3, 10, 11, 13, 17, 18, 35, 36],
        13: [11, 12, 14, 17, 18, 35, 36],
        14: [0, 6, 22, 24, 35, 36, 13],
        15: [7, 8, 9, 16, 36],
        16: [7, 8, 9, 10, 11, 15, 17, 36],
        17: [10, 11, 12, 13, 16, 18, 36],
        18: [11, 12, 13, 17, 35, 36],
        19: [20],
        20: [19, 23, 26],
        21: [27],
        22: [0, 14, 24, 35],
        23: [20],
        24: [14, 22, 28, 34, 35, 0],
        25: [26, 29, 30],
        26: [20, 25, 27, 29, 30],
        27: [21, 26, 31, 32, 35],
        28: [24, 33, 34, 35],
        29: [25, 26, 30],
        30: [25, 26, 29],
        31: [27, 32],
        32: [27, 31],
        33: [28, 34],
        34: [24, 28, 33],
        35: [0, 6, 11, 12, 13, 14, 18, 22, 24, 27, 28, 34],
        36: [2, 3, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18],
    }
    
    t_m = start_matrix
    best_score, best_tm = cost_t_matrix(start_matrix), start_matrix
    print(f"Starting process. Initial score = {best_score}")

    # For each iteration, try to adjust a single transition probability, 
    # by increasing it by step_size (the staying probability remains the same, however)
    for _ in range(n_iterations):
        start = np.random.choice([K for K, V in neighboursDict.items() if len(V) > 0])
        end = np.random.choice(neighboursDict[start])
        t_m[start][end] += step_size
        t_m[start][start] -= step_size

        new_cost = cost_t_matrix(t_m)
        if new_cost < best_score:
            best_score, best_tm = new_cost, t_m
        else:
            t_m[start][end] -= step_size
            t_m[start][start] += step_size
            neighboursDict[start].remove(end)
            if start in neighboursDict[end]:
                neighboursDict[end].remove(start)
            print(f"Disconnected {start} from {end}")
            
        print(f"Done an iteration, new best = {best_score}")

    end_time = time.perf_counter()
    print(f"Took {end_time - start_time} seconds.")
    print(f"Best Score Attained: {best_score}")
    # print(f"Transition Matrix: \n{t_m}")
    print("New neighboursDict:")
    for K, V in neighboursDict.items():
        print(f"{K}: {V},")
