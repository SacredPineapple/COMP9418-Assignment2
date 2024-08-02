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

    for room, neighbours in neighboursDict.items():
        for n in neighbours:
            if room not in neighboursDict[n]:
                neighboursDict[n] += [room]

    transitions = {
        room: [(room, 0.8)] + [(n, 0.2 / len(neighbours)) for n in neighbours]
        for room, neighbours in neighboursDict.items()
    }

    return transitions