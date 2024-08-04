from gaussian_factor import GaussianFactor

class DoorSensor:
    # Create a door sensor between area1 and area2
    def __init__(self, area1, area2):
        self.area1 = area1
        self.area2 = area2

    # TODO: Assess the reliability of the sensors, using the training data.
    def update(self, data):
        if data == None:
            self.count = None
            return
        self.count = data

        # Uh
        self.vars = (0.1**2 * self.count**2) + 0.01

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars, prev_means, t_matrix):
        if self.count == None:
            return means, vars
        
        if (prev_means[self.area1] + prev_means[self.area2]) == 0:
            return means, vars
        
        # 1. Increase variance by some amount
        vars[self.area1] += self.vars
        vars[self.area2] += self.vars
        
        # 2. Adjust means according to prior proportion of people either side of door and transition probability
        # 2a Undo regular transition otherwise double counting effect
        delta1 = prev_means[self.area1] * t_matrix[self.area1,self.area2]
        delta2 = prev_means[self.area2] * t_matrix[self.area2,self.area1]
        
        new_mean1 = means[self.area1] - (delta2 - delta1)
        new_mean2 = means[self.area2] - (delta1 - delta2)
        
        # 2b Apply new transition with door
        new_mean1 += (delta2 - delta1)/(delta1 + delta2)*self.count
        new_mean2 += (delta1 - delta2)/(delta1 + delta2)*self.count
        
        # 2c Ensure non negative
        if new_mean1 < 0:
            new_mean2 += new_mean1
            new_mean1 = 0
            
        if new_mean2 < 0:
            new_mean1 += new_mean2
            new_mean2 = 0      
                      
        means[self.area1] = new_mean1
        means[self.area2] = new_mean2
        
        return means, vars