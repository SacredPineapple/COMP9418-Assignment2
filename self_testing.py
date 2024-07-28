import datetime
import numpy as np
import pandas as pd
import ast

# import the function written by the student
from solution import get_action

# simulator code
class Person:    
    def __init__(self, name, office=None):
        self.name = name
        
    def timestep(self, building_simulator):
        pass

class Motion:
    def __init__(self, name, room):
        self.room = room
        self.name = name
    def get_output(self, room_occupancy):
        pass

class Camera:
    def __init__(self, name, room):
        self.room = room
        self.name = name
    def get_output(self, room_occupancy):
        pass

class DoorSensor:
    def __init__(self, name, rooms):
        self.rooms = rooms #pair of rooms
        self.name = name
    def get_output(self, building_simulator):
        pass

class Robot:
    def __init__(self, name, start_room):
        self.name = name
        self.current_location = start_room
    def get_output(self, building_simulator):
        pass        
    def timestep(self, building_simulator):
        pass

class SelfTester:
    def __init__(self):
        self.data = pd.read_csv("data1.csv")
        self.curr_step = 0
        self.start_time = datetime.time(hour=8,minute=0)
        self.end_time = datetime.time(18,0)
        self.time_step = datetime.timedelta(seconds=15) # 15 seconds
        self.current_time = self.start_time
        self.room_occupancy = dict([(room, 0) for room in ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 
                                                           'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18', 'r19', 
                                                           'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 
                                                           'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'c1', 'c2', 'outside']])
        self.room_occupancy['outside'] = self.data.iloc[0]['outside']
        self.motion_sensors = [Motion('motion_sensor1',['r1']),Motion('motion_sensor2',['r14']),
                               Motion('motion_sensor3',['r19']),Motion('motion_sensor4',['r28']),
                               Motion('motion_sensor5',['r29']),Motion('motion_sensor6',['r32'])]
        
        self.door_sensors = [DoorSensor('door_sensor1',('r2','r3')),DoorSensor('door_sensor2',('c1','c2')),
                             DoorSensor('door_sensor3',('c1','r28')),DoorSensor('door_sensor4',('r20','r26'))]

        self.camera_sensors = [Camera('camera1','r3'), Camera('camera2','r21'),Camera('camera3','r25'),
                               Camera('camera4','r34')]
        
        self.robot_sensors = [Robot('robot1','r1'), Robot('robot2','r2')]

    def timestep(self):
        # get data for current timestep (this example test uses saved data instead of randomly simulated data)
        current_data = self.data.iloc[self.curr_step]
        
        # move people 
        for room in self.room_occupancy:
            self.room_occupancy[room] = current_data.loc[room]

        # increment time
        self.current_time = (datetime.datetime.combine(datetime.date.today(), self.current_time) + self.time_step).time()

        # work out sensor data
        sensor_data = {}
        for sensor in self.motion_sensors:
            sensor_data[sensor.name] = current_data[sensor.name]
        for sensor in self.camera_sensors:
            sensor_data[sensor.name] = int(current_data[sensor.name])
        for robot in self.robot_sensors:
            robot.timestep(self)
            sensor_data[robot.name] = ast.literal_eval(current_data[robot.name])
        for sensor in self.door_sensors:
            sensor_data[sensor.name] = int(current_data[sensor.name])

        # To make sure your code can handle this case,
        # set one random sensor to None
        broken_sensor = np.random.choice(list(sensor_data.keys())) 
        sensor_data[broken_sensor] = None

        sensor_data['time'] = self.current_time 

        self.curr_step += 1

        return sensor_data
    
simulator = SelfTester()
total_cost = 0

nEpochs = 10
for i in range(nEpochs):
    sensor_data = simulator.timestep()
    actions_dict = get_action(sensor_data)