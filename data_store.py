# Stores temporary handmade values for transition probs
import numpy as np

# # Temporary handmade transition probabilities
# # Key is start room, Value is list of (end room, probability,)
# neighbourMatrix = np.zeros((37, 37))

# # Make i and j neighbours
# def make_neighbours(i, j):
#     neighbourMatrix[i, j] = 1
#     neighbourMatrix[j, i] = 1

# pairs = [
#     (1, 2),
#     (2, 3),
#     (3, 12),
#     (4, 5),
#     (4, 6),
#     (5, 6),
#     (6, 14),
#     (7, 8),
#     (7, 15),
#     (7, 36),
#     (8, 15),
#     (8, 36),
#     (8, 9),
#     (9, 10),
#     (9, 15),
#     (9, 16),
#     (9, 36),
#     (10, 11),
#     (10, 16),
#     (10, 36),
#     (11, 12),
#     (11, 17),
#     (11, 18),
#     (11, 36),
#     (12, 13),
#     (12, 17),
#     (12, 18),
#     (12, 36),
# ]
# for i, j in pairs:
#     make_neighbours(i, j)

# Note: I gave up

def getTransitions():
    neighboursDict = {
        0: [14, 22, 24, 28, 35, 36],
        1: [2, 3, 12],
        2: [1, 3, 12, 36],
        3: [1, 2, 12],
        4: [6, 14],
        5: [6, 14],
        6: [4, 5, 14, 22, 35, 36],
        7: [16, 36],
        8: [15, 36],
        9: [10, 16, 36],
        10: [9, 11, 12, 16, 36],
        11: [10, 36],
        12: [1, 2, 3, 10, 16, 35, 36],
        13: [9, 36],
        14: [0, 6, 22, 24, 35, 36],
        15: [8, 12, 22, 36],
        16: [7, 9, 10, 12, 14, 35, 36],
        17: [36],
        18: [7, 36],
        19: [20],
        20: [19, 23, 26, 35],
        21: [27, 32, 35],
        22: [6, 14, 24, 28, 35],
        23: [20, 26],
        24: [22, 28, 35],
        25: [20, 26, 27, 29, 30],
        26: [20, 25, 27, 29, 30, 35],
        27: [21, 26, 32, 35],
        28: [22, 24, 34, 35],
        29: [25, 26, 30],
        30: [25, 26, 29],
        31: [27, 32],
        32: [27, 31, 35],
        33: [34],
        34: [24, 28, 33, 35],
        35: [3, 6, 12, 13, 14, 22, 24, 27, 28],
        36: [2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22],
    }

    for room, neighbours in neighboursDict.items():
        for n in neighbours:
            if room not in neighboursDict[n]:
                neighboursDict[n] += [room]

    transitions = {
        room: [(room, 0.8)] + [(n, 0.2 / len(neighbours)) for n in neighbours] 
        for room, neighbours in neighboursDict.items()
    }

    return transitions