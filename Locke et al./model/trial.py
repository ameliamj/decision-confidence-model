# used to store the raw data of a trial
# has all fields necessary for creating the model assuming a uniform prior and value distribution
# for description of data fields see: https://osf.io/e2zrn
class Trial:
    def __init__(self, subject, stimulus, r1, r2):
        self.subject = subject
        self.stimulus = stimulus
        self.reaction = r1
        self.conf = r2

    # returns true if the prior distribution of probabilities is uniform
    def evenPrior(self):
        return True

    def evenReward(self):
        return True

# trial object with relevant data fields for the model in the case of
# non-uniform distributions of the prior or value
# prior: the prior probability distribution in the form [pR, pL]
class PriorTrial(Trial):
    def __init__(self, subject, session, prior, value, trial_num, stimulus, r1, r2):
        super().__init__(subject, stimulus, r1, r2)
        self.value = value
        self.prior = prior
        self.session = session
        self.trial_num = trial_num

    # returns true if the prior distribution of probabilities is uniform
    def evenPrior(self):
        return self.prior[0] == self.prior[1]

    def evenReward(self):
        return self.value[0] == self.value[1]
    
    def fullySymmetric(self):
        return self.evenReward() and self.evenPrior()

    def __str__(self):
        return '[' + self.subject + ', ' + self.session + ', ' + self.stimulus + ']'
