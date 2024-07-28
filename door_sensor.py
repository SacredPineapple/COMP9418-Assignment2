from gaussian_factor import GaussianFactor

class DoorSensor:
    # Create a door sensor between area1 and area2
    def __init__(self, area1, area2):
        self.area1 = area1
        self.area2 = area2

    # TODO: Assess the reliability of the sensors, using the training data.
    def update(self, data):
        self.count = data

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars):
        if self.count == None:
            return means, vars
        return means, vars