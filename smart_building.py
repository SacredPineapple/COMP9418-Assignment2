### The central class that manages the network of the PGM (i.e. each room and its associated connections)
import numpy as np
import scipy.sparse as sp

from sensors.camera_sensor import CameraSensor
from sensors.door_sensor import DoorSensor
from sensors.motion_sensor import MotionSensor
from sensors.robot_sensor import RobotSensor

from data_store import transitions

class SmartBuilding:
    def __init__(self):
        # outside has index 0, r1-34 have index 1-34, c1 and c2 have index 35, 36
        # Helper maps. Convention is that internally, we use the indexes (numbers) for everything,
        # and only convert it back to names when returning a query.
        self.nAreas = 37
        self.idx_to_name = ['outside'] + ['r' + str(i) for i in range(1, 35)] + ['c1', 'c2']
        self.name_to_idx = {'outside': 0} | {'r' + str(i): i for i in range(1, 35)} | {'c1': 35, 'c2': 36}

        # TODO: Learned parameters for transition probabilities will be initialised here, once this takes in a 'params' argument
        # TODO: Transition matrices that changes by time/phase of day?
        #   - Early morning entering office
        #   - Morning work block
        #   - Lunch hour
        #   - Afternoon work block
        #   - Late afternoon leaving office
        t_m = sp.csr_array((self.nAreas, self.nAreas))
        for i, j, k in transitions:
            t_m[i, j] = k
        self.t_matrix = t_m
        # Element-wise square, for the variance transitions
        self.t_matrix_sq = np.square(self.t_matrix)

        # Initial state is a N(40, 9) distribution outside
        # We store the state as follows:
        #   - The distribution of people in each area is assumed to be Gaussian
        #   - state_means stores the mean of each Gaussian for each area
        #   - state_vars stores the variance of each Gaussian for each area
        #
        # Transitions are assumed to work as below, with an example for room 24:
        # mu_24 = t24_24*mu_p24 + t22_24*mu_p22 + t28_24*mu_p28
        # var_24 = t24_24^2*var_p24 + t22_24^2*var_p22 + t28_24^2*var_p28
        #  
        # NOTE: This means that variance decays towards zero over time. Makes sense for a normal transition 
        # matrix as we're approaching the limiting state, however of course this isn't really a perfect
        # Markov process. I suppose evidence variables can bring the variance back up?
        self.state_means = np.array([40] + [0]*(self.nAreas - 1))
        self.state_vars = np.array([9] + [0.25]*(self.nAreas - 1))

        # Initialise sensors that will be used for evidencing.
        self.sensors = {
            'motion_sensor1': MotionSensor(1),
            'motion_sensor2': MotionSensor(14),
            'motion_sensor3': MotionSensor(19),
            'motion_sensor4': MotionSensor(28),
            'motion_sensor5': MotionSensor(29),
            'motion_sensor6': MotionSensor(32),
            'door_sensor1': DoorSensor(2, 3),
            'door_sensor2': DoorSensor(35, 36),
            'door_sensor3': DoorSensor(20, 26),
            'door_sensor4': DoorSensor(28, 35),
            'camera1': CameraSensor(3),
            'camera2': CameraSensor(21),
            'camera3': CameraSensor(25),
            'camera4': CameraSensor(34),
            'robot1': RobotSensor(),
            'robot2': RobotSensor(),
        }
    
    ### Increment one tick (15 seconds)
    def tick(self):
        # Adjust the state_means and state_vars as by the transition matrix.
        self.state_means = self.state_means @ self.t_matrix
        self.state_vars = self.state_vars @ self.t_matrix_sq
    
    ### Incorporate the evidence from the sensor data to the current model. 
    def apply_evidence(self, sensor_data):
        # Each sensor is independent (we're assuming), so we can, for each sensor:
        # Create a factor for that sensor, join it in, evidence along that factor.
        # TODO: Are the sensors all independent? For instance, r3 camera + door sensor would have ties.
        for sensor_name, data in sensor_data.items():
            if sensor_name in self.sensors.keys():
                self.sensors[sensor_name].update(data)
                self.state_means, self.state_vars = self.sensors[sensor_name].apply_evidence(self.state_means, self.state_vars)
    
    ### Query the network to return the normal distributions representing each room's occupancy.
    def query(self):
        area_params = [{'mean': mean, 'var': var} for mean, var in zip(self.state_means, self.state_vars)]
        return {self.idx_to_name[i]: area_params[i] for i in range(self.nAreas)}
    
