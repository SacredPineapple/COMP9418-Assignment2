from gaussian_factor import GaussianFactor

class MotionSensor:
    # Create a MotionSensor active in specified area
    def __init__(self, area):
        self.area = area

    # TODO: Assess the reliability of the sensors, using the training data.
    def update(self, data):
        self.response = data

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars):
        if self.response == None:
            return means, vars
        
        if self.response == 'no motion':
            ...
        elif self.response == 'motion':
            ...
        
        return means, vars