from gaussian_factor import GaussianFactor

class CameraSensor:
    # Create a CameraSensor active in specified area
    def __init__(self, area):
        self.area = area

    # Assuming here that the variance of the CameraSensor count is increasing in the count.
    # TODO: Assess the reliability of the sensors. Do the sensors have a bias, is the above assumption correct, etc.
    def update(self, data):
        if data == None:
            self.count = None
            return
        self.count = data

        # 95% of the time the true measure is within 0.7-1.3 times the mean i.e. 1.s.d = 0.15*mean)
        # Using the observed value as our guess for the true mean.
        self.vars = (0.1**2 * self.count**2) + 0.01

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars):
        if self.count == None:
            return means, vars
        # Create a temporary mini-Bayesian network:
        # [ ppl_count ] -> [ obs_ppl_count ]
        #
        # We are seeking to find the updated distribution:
        # ppl_count | obs_ppl_count = self.count
        prior = GaussianFactor(('num_ppl',), mu=means[self.area], sigma=vars[self.area])
        camera = GaussianFactor(('obs_num_ppl', 'num_ppl',), beta=[1], b_mean=0, b_var=self.vars)
        joint = prior * camera
        final = joint.evidence(obs_num_ppl=self.count)

        # Error checking - we should have eliminated all but 1 variable now, so mean and variance should be 1d
        # If this assertion passes, extract the sole element as a constant
        assert(len(final.mean()) == 1)
        assert(len(final.covariance()) == 1)
        means[self.area] = (final.mean()).reshape(1)[0]
        vars[self.area] = (final.covariance()).reshape(1)[0]

        return means, vars