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
import scipy as sp
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

###################################
# Code stub
#
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
#


# this global state variable demonstrates how to keep track of information over multiple
# calls to get_action
state = {}

# cost constants
COST_LIGHT = 1
COST_PERSON_NO_LIGHT = 4

# Initialising classes, variables, etc.
# params = pd.read_csv(...)
# pgm = ProbabilisticGraphicalModel(params) (idk)

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state
    #global params

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data

    # Step 1: Pass in the sensor data to the PGM, to set evidence
    # pgm.update_data(sensor_data)
    
    # Step 2: Query the pgm to get the expected number of people in each area? Maybe?
    # info = pgm.query()

    # Step 3: Use this expected num. people per area to create and return our actions_dict
    # return info_to_actions(info)

    # Other considerations
    # 1. We might want to incorporate things other than just the sensor data into our model
    #    e.g. time of day, certain movement behaviours and patterns, hypothesised room functions
    # 2. Maybe needs more classes? Something to process the sensor data into a form which 
    #    can just be passed to the pgm.
    # 3. Uh is this a Gaussian Model

    # actions_dict = {'lights1': 'on', 'lights2': 'on', 'lights3': 'on', 'lights4': 'on',
    #                 'lights5': 'on', 'lights6': 'on', 'lights7': 'on', 'lights8': 'on',
    #                 'lights9': 'on', 'lights10': 'on', 'lights11': 'on', 'lights12': 'on',
    #                 'lights13': 'on', 'lights14': 'on', 'lights15': 'on', 'lights16': 'on',
    #                 'lights17': 'on', 'lights18': 'on', 'lights19': 'on', 'lights20': 'on',
    #                 'lights21': 'on', 'lights22': 'on', 'lights23': 'on', 'lights24': 'on',
    #                 'lights25': 'on', 'lights26': 'on', 'lights27': 'on', 'lights28': 'on',
    #                 'lights29': 'on', 'lights30': 'on', 'lights31': 'on', 'lights32': 'on',
    #                 'lights33': 'on', 'lights34': 'on'}
    
    # Start with filling initial state if this is the first visit;
    # Note that the spec says num_people = round(Normal(mean=40, stddev=3)), 
    # TODO: for now just setting it to always be 40 people but probably need to somehow shift with evidence later?
    if state == {}:
        state = {'r1': 0, 'r2': 0, 'r3': 0, 'r4': 0, 'r5': 0, 'r6': 0, 'r7': 0, 'r8': 0, 'r9': 0, 
                 'r10': 0, 'r11': 0, 'r12': 0, 'r13': 0, 'r14': 0, 'r15': 0, 'r16': 0, 'r17': 0, 
                 'r18': 0, 'r19': 0, 'r20': 0, 'r21': 0, 'r22': 0, 'r23': 0, 'r24': 0, 'r25': 0, 
                 'r26': 0, 'r27': 0, 'r28': 0, 'r29': 0, 'r30': 0, 'r31': 0, 'r32': 0, 'r33': 0,
                 'r34': 0, 'outside': 40}
    
    # Transition parameters to learn with training data?
    p = {'o22': 0.2, '22o': 0.1}
    
    # Setting up GaussianFactors
    # p+name means previous state of that room
    r22 = GaussianFactor(('r22', 'pr22', 'po'), beta=[(1-p['22o']), p['o22']], b_mean=0, b_var=0.5**2)
    o = GaussianFactor(('o', 'po', 'pr22'), beta=[(1-p['o22']), p['22o']], b_mean=0, b_var=0.5**2)
    
    model = o.join(r22)
    
    # Predict people in room 22 and outside
    new_state = state
    
    # Simulate normal distributions and sample new state values
    emodel = model.evidence(po=state['outside'], pr22=state['r22'])
    
    new_state['r22'] = emodel.sample()['r22']
    new_state['outside'] = emodel.sample()['o']
    
    state = new_state
    
    return info_to_actions(state)

# Presumably, we're expecting an output of the expected number of people in each room.
# If there's more than or equal to 0.25 people in the room, turn the light on.
# Right now, probs accepts a dictionary of form {'r1': 0.25, 'r2': 0.3, 'r3': 0.1, 'r4': 1, ..., 'c1': 2, 'c2': 0.2, 'outside': '11'}
# and returns a dictionary of form {'lights1': 'on', 'lights2': 'on', ..., 'lights34': 'off'}
def info_to_actions(info):
    actions_dict = {'lights' + str(match.group(1)) : 'off' if nPeople < COST_LIGHT / COST_PERSON_NO_LIGHT else 'on'
                    for area, nPeople in info.items() if (match := re.match(r'r(\d+)', area))}
    return actions_dict