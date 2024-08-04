from gaussian_factor import GaussianFactor

class DoorSensor:
    # Create a door sensor between area1 and area2
    def __init__(self, area1, area2):
        self.bias = 0
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
    # This is a pre-tick sensor - it takes in the means and vars of the previous
    # tick, and updates our beliefs about them. This step happens before
    # the transition matrix.
    def apply_evidence(self, means, vars, t_m):
        if self.count == None:
            return means, vars
        
        # We locally pre-empt the learned transition probabilities given the door sensor data
        # self.count comes from: movement from 1->2 + movement from 2->1
        #
        # Example scenario:
        # If previously there were 100 people in area 1, 10 people in area 2, and we have transition probabilities
        # 1->2: 10%, 2->1: 20%, we get (100*10%) + (10*20%) = 12 as our expected (mean) door sensor reading.
        #
        # We can create a Gaussian factor for the door sensor, mean
        # (mu1 * t_1-2) + (mu2 * t_2-1), variance as determined from data.
        # Evidencing on this will give us new estimates for the (previous) mean and variance.
        prev_area1 = GaussianFactor(('num_ppl1',), mu=means[self.area1], sigma=vars[self.area1])
        prev_area2 = GaussianFactor(('num_ppl2',), mu=means[self.area2], sigma=vars[self.area2])
        transitions = [t_m[self.area1, self.area2], t_m[self.area2, self.area1]]
        door_sensor = GaussianFactor(('door_reading', 'num_ppl1', 'num_ppl2',), beta=transitions, b_mean=self.bias, b_var=self.vars)

        joint = door_sensor * prev_area1 * prev_area2
        final = joint.evidence(door_reading=self.count)

        # print(final.mean())
        # print(final.covariance())

        means[self.area1], means[self.area2] = final.mean()[0], final.mean()[1]
        vars[self.area1], vars[self.area2] = final.covariance()[0, 0], final.covariance()[1, 1]

        return means, vars
        
        # # 1. Increase variance by some amount
        # vars[self.area1] += self.vars
        # vars[self.area2] += self.vars
        
        # # 2. Adjust means according to prior proportion of people either side of door and transition probability
        # # 2a Undo regular transition otherwise double counting effect
        # delta1 = prev_means[self.area1] * t_matrix[self.area1,self.area2]
        # delta2 = prev_means[self.area2] * t_matrix[self.area2,self.area1]
        
        # new_mean1 = means[self.area1] - (delta2 - delta1)
        # new_mean2 = means[self.area2] - (delta1 - delta2)
        
        # # 2b Apply new transition with door
        # new_mean1 += (delta2 - delta1)/(delta1 + delta2)*self.count
        # new_mean2 += (delta1 - delta2)/(delta1 + delta2)*self.count
        
        # # 2c Ensure non negative
        # if new_mean1 < 0:
        #     new_mean2 += new_mean1
        #     new_mean1 = 0
            
        # if new_mean2 < 0:
        #     new_mean1 += new_mean2
        #     new_mean2 = 0      
        
        # means[self.area1] = new_mean1
        # means[self.area2] = new_mean2
        
        # return means, vars