import numpy as np

from gaussian_factor import GaussianFactor

class DoorSensor:
    # Create a door sensor between area1 and area2
    def __init__(self, area1, area2):
        self.bias = 0
        self.area1 = area1
        self.area2 = area2

    def update(self, data):
        if data == None:
            self.count = None
            return
        self.count = data

    # Apply evidence on the room distributions given the currently stored evidence.
    # This is a pre-tick sensor - it takes in the means and vars of the previous
    # tick, and updates our beliefs about them. This step happens before the transition matrix.
    def apply_evidence(self, means, vars, t_m):
        if self.count == None:
            return means, vars
        
        # Example scenario:
        # If previously there were 100 people in area 1, 10 people in area 2, and we have transition probabilities
        # 1->2: 10%, 2->1: 20%, we get (100*10%) + (10*20%) = 12 as our expected (mean) door sensor reading.
        #
        # We can create a Gaussian factor for the door sensor, mean
        # (mu1 * t_1-2) + (mu2 * t_2-1), variance proportional to the mean
        # Evidencing on this will give us new estimates for the (previous) mean and variance.
        prev_area1 = GaussianFactor(('num_ppl1',), mu=means[self.area1], sigma=vars[self.area1])
        prev_area2 = GaussianFactor(('num_ppl2',), mu=means[self.area2], sigma=vars[self.area2])
        transitions = np.array([t_m[self.area1, self.area2], t_m[self.area2, self.area1]])
        b_variance = np.abs(0.216 * transitions @ np.array([means[self.area1], means[self.area2]])) + 0.0001
        door_sensor = GaussianFactor(('door_reading', 'num_ppl1', 'num_ppl2',), beta=transitions, b_mean=self.bias, b_var=b_variance)

        joint = door_sensor * prev_area1 * prev_area2
        final = joint.evidence(door_reading=self.count)

        means[self.area1], means[self.area2] = final.mean()[0], final.mean()[1]
        vars[self.area1], vars[self.area2] = final.covariance()[0, 0], final.covariance()[1, 1]

        return means, vars