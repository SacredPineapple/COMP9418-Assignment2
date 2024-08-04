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
with open("redistributed.pkl", "rb") as f:
    twoHourlyTransitions = pickle.load(f)

smart_building = SmartBuilding(twoHourlyTransitions)

def get_action(sensor_data):
    # Step 1: Increment the network to the next step, incorporating the sensor data.
    smart_building.tick(sensor_data)

    # Step 2: Query the model to get the distribution of the number of people in each room
    means, vars = smart_building.query()

    # Step 3: Convert each room's distribution into a decision about turning light on/off
    actions = info_to_actions(means, vars)

    return actions

# Accepts an input of form np.array([r1_mean, r2_mean, ...]), np.array([r1_vars, r2_vars, ...])
# Returns dictionary of form {'lights1': 'on', 'lights2': 'on', ..., 'lights34': 'off'}
# Compares the expected cost of turning a light on (COST_LIGHT) vs. turning a light off (some integral) to determine action.
# Not intended to be very readable - the integral form was simplified into another form, which is what is used here 
def info_to_actions(mus, vars):
    sigmas = np.sqrt(vars)
    pow = - mus**2 / (2*vars)
    term1 = ( sigmas / np.sqrt(2*np.pi) ) * np.exp(pow)
    term2 = mus * scipy.stats.norm.sf(0, loc=mus, scale=sigmas)
    cost_light_off = (term1 + term2) * COST_PERSON_NO_LIGHT

    light_on = cost_light_off > COST_LIGHT
    actions_dict = {'lights' + str(i+1): 'on' if light_on[i] else 'off' for i in range(34)}
    return actions_dict
