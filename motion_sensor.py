import numpy as np

from gaussian_factor import GaussianFactor

class MotionSensor:
    # Create a MotionSensor active in specified area
    def __init__(self, area):
        self.area = area

    def update(self, data):
        self.response = data

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars, t_m):
        if self.response == None:
            return means, vars
        
        observation = means[self.area]
        if self.response == 'no motion':
            # If we get no motion, we convert this into an observation that there were 0 people in the room.
            observation = 0
        elif self.response == 'motion':
            # If we get yes motion, we convert this into an observation that there were people in the room
            # We only slightly skew up our belief if our existing belief was already positive,
            # but we skew up our belief more significantly if our existing belief was close to 0
            #
            # A nice way to do this is by skewing up using a decreasing function: lambda * exp(-{existing belief})
            observation = means[self.area] + 0.5 * np.exp(-means[self.area])
        
        prior = GaussianFactor(('num_ppl',), mu=means[self.area], sigma=vars[self.area])
        motion = GaussianFactor(('obs_num_ppl', 'num_ppl',), beta=[1], b_mean=0, b_var=0.09)
        joint = prior * motion
        final = joint.evidence(obs_num_ppl=observation)

        means[self.area] = (final.mean()).reshape(1)[0]
        vars[self.area] = (final.covariance()).reshape(1)[0]
        
        return means, vars