'''
COMP9418 Assignment 2

Name: Jordan Huang    zID: z5418948
Name: Sam Hodgson    zID: z5416863

'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle
import json

from gaussian_factor import GaussianFactor
from smart_building import SmartBuilding

# Provided Code Stub - for submission testing, only get_action will be called.
#
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict

# Cost constants
COST_LIGHT = 1
COST_PERSON_NO_LIGHT = 4

# Initialising classes, variables, etc.
twoHourlyTransitions = np.load("avgTransitionRedistributed5double.npy")
print(f"Loading in transition matrix: has dimensions {twoHourlyTransitions.shape}")
smart_building = SmartBuilding(twoHourlyTransitions)

step = 0
true_state = None
def update_true_state(data):
    global true_state
    true_state = data

def get_action(sensor_data):
    # Step 1: Increment the network to the next step, incorporating the sensor data.
    smart_building.tick(sensor_data)

    # Step 2: Query the model to get the distribution of the number of people in each room
    means, vars = smart_building.query()

    # Step 3: Convert each room's distribution into a decision about turning light on/off
    actions = info_to_actions(means, vars)

    # Printing for debugging / seeing whats happening
    # global step
    # if step % 500 == 0:
    #     print(f"----------------Time {step}----------------")
    #     for i in range(34):
    #         mu, sigma = means[i], np.sqrt(vars[i])
    #         value = true_state['r'+str(i+1)]
    #         zscore = (value - mu) / sigma
    #         light_status = actions['lights' + str(i+1)]

    #         str1 = f"Room {i+1}: {value} ~ N({round(mu, 4)}, {round(sigma, 4)}): "
    #         str2 = f"{'ON' if light_status == 'on' else 'OFF'} "
    #         str3 = f"z-score: {round(zscore, 4)}"
    #         print(str1 + str2 + str3)
    #         if value > 0 and light_status == 'off':
    #             print("NORMAL DIST INSUFFICIENT HERE - OVERCONFIDENT")
    #         if value == 0 and light_status == 'on':
    #             print("TURNED LIGHT ON FOR NO REASON RIP")
    # step += 1

    return actions

# Accepts an input of form np.array([r1_mean, r2_mean, ...]), np.array([r1_vars, r2_vars, ...])
# Returns dictionary of form {'lights1': 'on', 'lights2': 'on', ..., 'lights34': 'off'}
# Compares the expected cost of turning a light on (COST_LIGHT) vs. turning a light off (some integral) to determine action.
# Math was used to determine the integral formula, and is used here (maybe put this in the report?)
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
