### The central class that manages the network of the PGM (i.e. each room and its associated connections)
import numpy as np
import scipy.sparse as sp
import datetime

from camera_sensor import CameraSensor
from door_sensor import DoorSensor
from motion_sensor import MotionSensor
from robot_sensor import RobotSensor

class SmartBuilding:
    def __init__(self, transition_matrix):
        # outside has index 0, r1-34 have index 1-34, c1 and c2 have index 35, 36
        # Helper maps. Convention is that internally, we use the indexes (numbers) for everything,
        # and only convert it back to names when returning a query.
        self.nAreas = 37
        self.idx_to_name = ['outside'] + ['r' + str(i) for i in range(1, 35)] + ['c1', 'c2']
        self.name_to_idx = {'outside': 0} | {'r' + str(i): i for i in range(1, 35)} | {'c1': 35, 'c2': 36}
        self.min_vars = 0.0001

        # TODO: Transition matrices that changes by time/phase of day?
        #   - Early morning entering office
        #   - Morning work block
        #   - Lunch hour
        #   - Afternoon work block
        #   - Late afternoon leaving office
        # t_m = sp.csr_array((self.nAreas, self.nAreas))
        self.t_matrix = transition_matrix
        # Element-wise square, for the variance transitions
        self.t_matrix_sq = self.t_matrix * self.t_matrix

        # Initial state is a N(40, 9) distribution outside
        # We store the state as follows:
        #   - The distribution of people in each area is assumed to be Gaussian
        #   - state_means stores the mean of each Gaussian for each area
        #   - state_vars stores the variance of each Gaussian for each area
        #
        # Transitions are assumed to work as below, with an example for room 24:
        # If X_24 = t24_24*Xp24 + t22_24*Xp22 + t28_24*Xp28
        # mu_24 = t24_24*mu_p24 + t22_24*mu_p22 + t28_24*mu_p28
        # var_24 = t24_24^2*var_p24 + t22_24^2*var_p22 + t28_24^2*var_p28
        #  
        # NOTE: This means that variance decays towards zero over time. Makes sense for a normal 
        # Markov chain as we're approaching the limiting state, however of course this isn't really a perfect
        # Markov process. I suppose evidence variables can bring the variance back up?
        self.state_means = np.array([40] + [0]*(self.nAreas - 1))
        self.state_vars = np.array([9] + [self.min_vars]*(self.nAreas - 1))

        # Initialise sensors that will be used for evidencing.
        self.post_tick_sensors = {
            'motion_sensor1': MotionSensor(1),
            'motion_sensor2': MotionSensor(14),
            'motion_sensor3': MotionSensor(19),
            'motion_sensor4': MotionSensor(28),
            'motion_sensor5': MotionSensor(29),
            'motion_sensor6': MotionSensor(32),
            'camera1': CameraSensor(3),
            'camera2': CameraSensor(21),
            'camera3': CameraSensor(25),
            'camera4': CameraSensor(34),
            'robot1': RobotSensor(),
            'robot2': RobotSensor(),
        }
        self.pre_tick_sensors = {
            'door_sensor1': DoorSensor(2, 3),
            'door_sensor2': DoorSensor(35, 36),
            'door_sensor3': DoorSensor(20, 26),
            'door_sensor4': DoorSensor(28, 35),
        }

    ### Increment one tick (15 seconds)
    def tick(self, sensor_data):
        # Adjust the state_means and state_vars as by the transition matrix.
        # Variance decays a little from its previous value, but also increases per tick based on 
        # the uncertainty of movement, proportional to the amount of movement experienced.
        
        # Evidence on door sensors first, as they give us information on the previous tick's distribution,
        # rather than the current one
        for sensor_name, sensor in self.pre_tick_sensors.items():
            sensor.update(sensor_data[sensor_name])
            sensor.apply_evidence(self.state_means, self.state_vars, self.t_matrix)

        nine_am_datetime = datetime.datetime.combine(datetime.date.today(), datetime.time(9, 0, 15))
        full_time_datetime = datetime.datetime.combine(datetime.date.today(), sensor_data['time'])
        delta = full_time_datetime - nine_am_datetime
        self.time_idx = delta.seconds // 3600 + delta.days * 24 + (delta.seconds > 0)
        self.time_idx = self.time_idx // 2
        
        self.state_means = self.state_means @ self.t_matrix[self.time_idx]
        self.state_vars = self.state_vars @ self.t_matrix_sq[self.time_idx] + 0.75 * (self.state_means ** 2)

        # Evidence on the remaining sensors
        for sensor_name, sensor in self.post_tick_sensors.items():
            sensor.update(sensor_data[sensor_name])
            sensor.apply_evidence(self.state_means, self.state_vars, self.t_matrix)

        # After applying evidence, normalise. This 'propagates' the evidence throughout the whole network.
        # For instance, if our evidence suggests there are a larger than expected number of people in a room than before,
        # normalising down again reduces the expected number of people elsewhere.
        self._normalize()
    
    ### Incorporate the evidence from the sensor data to the current model. 
    # Archived method, no longer used.
    def apply_evidence(self, sensor_data):
        # Each sensor is independent (we're assuming), so we can, for each sensor:
        # Create a factor for that sensor, join it in, evidence along that factor.
        # TODO: Are the sensors all independent? For instance, r3 camera + door sensor would have ties.
        self.state_vars = np.maximum(self.state_vars, 0.01)
        for sensor_name, data in sensor_data.items():
            if sensor_name in self.sensors.keys():
                self.sensors[sensor_name].update(data)
                
                if sensor_name.startswith("door"):
                    self.state_means, self.state_vars = self.sensors[sensor_name].apply_evidence(self.state_means, self.state_vars, self.prev_means, self.t_matrix[self.time_idx])
                else:
                    self.state_means, self.state_vars = self.sensors[sensor_name].apply_evidence(self.state_means, self.state_vars)
        
        # After applying evidence, normalise. This 'propagates' the evidence throughout the whole network.
        # For instance, if our evidence suggests there are a larger than expected number of people in a room than before,
        # normalising down again reduces the expected number of people elsewhere.
        self._normalize()

    def _normalize(self):
        norm_const = 40 / sum(self.state_means)
        self.state_means = norm_const * self.state_means
        self.state_vars = (norm_const ** 2) * self.state_vars
    
    ### Query the network to return the normal distributions representing each room's occupancy.
    def query(self):
        # Avoiding div-by-zero errors, ensuring variance is not 0
        self.state_vars = np.maximum(self.state_vars, self.min_vars)

        return self.state_means[1:35], self.state_vars[1:35]
    
