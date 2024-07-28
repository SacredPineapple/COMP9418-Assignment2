# Stores temporary handmade values for transition probs
import numpy as np

# Temporary handmade transition probabilities
# Key is start room, Value is list of (end room, probability,)
neighbourMatrix = np.zeros((37, 37))

# Make i and j neighbours
def make_neighbours(i, j):
    neighbourMatrix[i, j] = 1
    neighbourMatrix[j, i] = 1

pairs = [
    (1, 2),
    (2, 3),
    (3, 12),
    (4, 5),
    (4, 6),
    (5, 6),
    (6, 14),
    (7, 8),
    (7, 15),
    (7, 36),
    (8, 15),
    (8, 36),
    (8, 9),
    (9, 10),
    (9, 15),
    (9, 16),
    (9, 36),
    (10, 11),
    (10, 16),
    (10, 36),
    (11, 12),
    (11, 17),
    (11, 18),
    (11, 36),
    (12, 13),
    (12, 17),
    (12, 18),
    (12, 36),
]
for i, j in pairs:
    make_neighbours(i, j)

# Note: I gave up

transitions = {
    0: [(0, 0.8), (22, 0.2), ],
    1: [(1, 0.8), (2, 0.1), ],
    2: [(1, ), ],
    3: [(, ), ],
    4: [(, ), ],
    5: [(, ), ],
    6: [(, ), ],
    7: [(, ), ],
    8: [(, ), ],
    9: [(, ), ],
    10: [(, ), ],
    11: [(, ), ],
    12: [(, ), ],
    13: [(, ), ],
    14: [(, ), ],
    15: [(, ), ],
    16: [(, ), ],
    17: [(, ), ],
    18: [(, ), ],
    19: [(, ), ],
    20: [(, ), ],
    21: [(, ), ],
    22: [(, ), ],
    23: [(, ), ],
    24: [(, ), ],
    25: [(, ), ],
    26: [(, ), ],
    27: [(, ), ],
    28: [(, ), ],
    29: [(, ), ],
    30: [(, ), ],
    31: [(, ), ],
    32: [(, ), ],
    33: [(, ), ],
    34: [(, ), ],
    35: [(, ), ],
    36: [(, ), ],
}