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
#

# Cost constants
COST_LIGHT = 1
COST_PERSON_NO_LIGHT = 4

# Initialising classes, variables, etc.
# params = pd.read_csv(...)
from data_store import getTransitions
t_m = np.zeros((37, 37))
for i, ns in getTransitions().items():
    for j, k in ns:
        t_m[i, j] = k
smart_building = SmartBuilding(t_m)

step = 0
true_state = None
def update_true_state(data):
    global true_state
    true_state = data

def get_action(sensor_data):
    # Step 1: Increment the network to the next step.
    smart_building.tick()

    # Step 2: Apply the evidence gained from the sensor data
    smart_building.apply_evidence(sensor_data)
    
    # Step 3: Query the pgm to get the distribution of the number of people in each room
    means, vars = smart_building.query()

    # Step 4: Convert each room's distribution into a decision about turning light on/off
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
    #         if np.abs(zscore) > 2:
    #             print("NORMAL DIST INSUFFICIENT HERE - OVERCONFIDENT")
    #         if value == 0 and light_status == 'on':
    #             print("TURNED LIGHT ON FOR NO REASON RIP")
    # step += 1

    return actions

    # Other considerations
    # 1. We might want to incorporate things other than just the sensor data into our model
    #    e.g. time of day, certain movement behaviours and patterns, hypothesised room functions

    actions_dict = {'lights1': 'on', 'lights2': 'on', 'lights3': 'on', 'lights4': 'on',
                    'lights5': 'on', 'lights6': 'on', 'lights7': 'on', 'lights8': 'on',
                    'lights9': 'on', 'lights10': 'on', 'lights11': 'on', 'lights12': 'on',
                    'lights13': 'on', 'lights14': 'on', 'lights15': 'on', 'lights16': 'on',
                    'lights17': 'on', 'lights18': 'on', 'lights19': 'on', 'lights20': 'on',
                    'lights21': 'on', 'lights22': 'on', 'lights23': 'on', 'lights24': 'on',
                    'lights25': 'on', 'lights26': 'on', 'lights27': 'on', 'lights28': 'on',
                    'lights29': 'on', 'lights30': 'on', 'lights31': 'on', 'lights32': 'on',
                    'lights33': 'on', 'lights34': 'on'}

    # Start with filling initial state if this is the first visit;
    # Note that the spec says num_people = round(Normal(mean=40, stddev=3)), 
    # TODO: for now just setting it to always be 40 people but probably need to somehow shift with evidence later?
    if state == {}:
        state = {'r1': 4, 'r2': 4, 'r3': 4, 'r4': 4, 'r5': 4, 'r6': 4, 'r7': 4, 'r8': 4, 'r9': 4, 
                 'r14': 4, 'r11': 4, 'r12': 4, 'r13': 4, 'r14': 0, 'r15': 4, 'r16': 4, 'r17': 4, 
                 'r18': 4, 'r19': 4, 'r20': 4, 'r21': 4, 'r22': 0, 'r23': 4, 'r24': 0, 'r25': 4, 
                 'r26': 4, 'r27': 4, 'r28': 0, 'r29': 4, 'r34': 4, 'r31': 4, 'r32': 4, 'r33': 4,
                 'r34': 4, 'c1': 0, 'c2': 0, 'outside': 40}
    
    # For now keeping simple and trying to predict just people in room 14, 22 and outside using 1 motion detector and made up transition parameters 
    # Non 0 transitions; outside is 0, c1 is 35 and c2 is 36
    # TODO: Learn from training data
    data = [(0, 22, 0.2), (14, 22, 0.1), (14, 35, 0.1), (22, 0, 0.1), (22, 14, 0.1), (22, 24, 0.1), (24, 22, 0.1), (24, 28, 0.1),
            (28, 24, 0.1), (28,35, 0.1), (35, 14, 0.2), (35, 28, 0.2)]  # (row, col, value)

    # Create a sparse matrix
    t_m = sp.csr_matrix(([value for _, _, value in data], ([row for row, _, _ in data], [col for _, col, _ in data])), shape=(37, 37))  # matrix shape
    
    room_var = 0.5**2
    door_var = 0.8**2
    
    # Door evidence
    # Before changing transition rates make factor; since uses of door should be linearly related to change in the number of people in each room and the proportion of ways people got to that room
    d4 = GaussianFactor(('d4', 'pr28', 'r28', 'pc1', 'c1'), beta = [(-t_m[35,28]/(t_m[35,28] + t_m[24,28])), (t_m[35,28]/(t_m[35,28] + t_m[24,28])), 
                                                                     (-t_m[28,35]/(t_m[28,35] + t_m[14,35])), (t_m[28,35]/(t_m[28,35] + t_m[14,35]))], b_mean=0, b_var=door_var)

    # Change transition rates as well
    # Chance someone going through d4 was going up to c4:
    pd4 = t_m[28,35] / (t_m[28,35]+t_m[35,28])

    if (sensor_data['door_sensor4'] != None):
        if (state['r28'] != 0):
            d4a = GaussianFactor(('d4a',), mu = sensor_data['door_sensor4'] * pd4, sigma = door_var)
            t_m[28, 35] = d4a.sample()['d4a']/state['r28']
        
        if (state['c1'] != 0):
            d4b = GaussianFactor(('d4b',), mu = sensor_data['door_sensor4'] * (1 - pd4), sigma = door_var)
            t_m[35, 28] = d4b.sample()['d4b']/state['c1']
    
    # Setting up GaussianFactors
    # p+name means previous state of that room
    o = GaussianFactor(('o', 'po', 'pr22'), beta=[(1-t_m[0, 22]), t_m[22, 0]], b_mean=0, b_var=room_var)
    r14 = GaussianFactor(('r14', 'pr14', 'pr22', 'prc1'), beta = [(1-t_m[14, 22]-t_m[14, 35]), t_m[22, 14], t_m[35, 14]], b_mean=0, b_var=room_var)
    r22 = GaussianFactor(('r22', 'pr22', 'po', 'pr14'), beta=[(1-t_m[22, 0]-t_m[22, 14]), t_m[0, 22], t_m[14, 22]], b_mean=0, b_var=room_var)
    r24 = GaussianFactor(('r24', 'pr24', 'pr22', 'pr28'), beta = [(1-t_m[24, 22]-t_m[24, 28]), t_m[22, 24], t_m[28, 24]], b_mean=0, b_var=room_var)
    r28 = GaussianFactor(('r28', 'pr28', 'pr24', 'prc1'), beta=[(1-t_m[28, 24]-t_m[28, 35]), t_m[24, 28], t_m[35, 28]], b_mean=0, b_var=room_var)
    c1 = GaussianFactor(('c1', 'pc1', 'pr14', 'pr28'), beta=[(1-t_m[35, 14]-t_m[35, 28]), t_m[14, 35], t_m[28, 35]], b_mean=0, b_var=room_var)
    
    model = o.join(r14).join(r22).join(r24).join(r28).join(c1).join(d4)
    
    # Simulate normal distributions and sample new state values
    # Probably neaten up with loop so dont have to individually type out all vals (code design and all that)
    emodel = model.evidence(po=state['outside'], pr22=state['r22'], pr14=state['r14'], pr24=state['r24'], pr28=state['r28'], pc1=state['c1'])
    
    if (sensor_data['door_sensor4'] != None):
        emodel = emodel.evidence(d4=sensor_data['door_sensor4'])
    
    # TODO: Simulate detector evidence (can be improved)
    # Right now just using to check if theres no-one in the room
    new_state = state
    
    if sensor_data['motion_sensor2'] == 'no motion':
        m2 = GaussianFactor(('r14',), mu=0.4, sigma=1)
        emodel *= m2
    else: 
        # TODO: If the room was initally empty, and we saw motion, what do we do?
        # new_state['r14'] = emodel.sample()['r14']
        ...
        
    if sensor_data['motion_sensor4'] == 'no motion':
        m4 = GaussianFactor(('r28',), mu=0.4, sigma=1)
        emodel *= m4
    else: 
        # TODO: If the room was initally empty, and we saw motion, what do we do?
        # new_state['r28'] = emodel.sample()['r28']
        ...
    
    new_state['r22'] = emodel.sample()['r22']
    new_state['r24'] = emodel.sample()['r24']
    new_state['r14'] = emodel.sample()['r14']
    new_state['r28'] = emodel.sample()['r28']
    new_state['outside'] = emodel.sample()['o']
    
    state = new_state
    
    return info_to_actions(state)

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
