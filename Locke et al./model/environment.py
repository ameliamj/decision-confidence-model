import numpy as np
from .agent import Agent


# the environment generates observations for the agent, calculates real and predicted accuracy, and calculates real
# and predicted rates of high confidence
# it has access to all data from all trials and the observational noise
class Environment:
    def __init__(self, trials, sdz):
        self.trials = trials
        self.sdz = sdz

    # generates observations given a list of trials, a multiplier, and an observational variance
    # returns a list of the multiplier number of observations generated for every trial
    @staticmethod
    def obsvs_trials(trials, mult, sigz):
        np.random.seed(451)
        obsvs = np.empty(len(trials) * mult)
        for i, trial in enumerate(trials):
            obsvs[mult * i: mult * (i + 1)] = (np.random.normal(trial.stimulus, sigz, size = mult))
        return obsvs

    # calculates the accuracy given the stimuli and the choices from the agent
    @staticmethod
    def calc_acc(stimuli, choices):
        correct = 0
        for i, stimulus in enumerate(stimuli):
            if stimulus == choices[i]:
                correct += 1
        return correct / len(stimuli)

    # calculates the predicted accuracy of a set of given trials for a subject that has the given observational variance
    @staticmethod
    def get_pred_acc(trials, sigz, sigz_sub, bias=0, mult = 100):
        agent = Agent(sigz_sub, bias)
        priors = np.zeros((len(trials) * mult, 2))
        values = np.zeros((len(trials) * mult, 2))
        stimuli = np.empty(len(trials) * mult)
        for i, trial in enumerate(trials):
            priors[i * mult : (i + 1) * mult] = np.tile(trial.prior, [mult, 1])
            values[i * mult : (i + 1) * mult] = np.tile(trial.value, [mult, 1])
            stimuli[i * mult : (i + 1) * mult] = trial.stimulus * np.ones(mult)


        obsvs = Environment.obsvs_trials(trials, mult, sigz)
        preds = agent.predict_choice(obsvs, priors, values)

        return Environment.calc_acc(stimuli, preds)

    # calculates the real accuracy of a subject for the given trials
    @staticmethod
    def get_real_acc(trials):
        stimuli = np.empty(len(trials))
        real_choices = np.empty(len(trials))
        for i, trial in enumerate(trials):
            stimuli[i] = trial.stimulus
            real_choices[i] = trial.reaction
        return Environment.calc_acc(stimuli, real_choices)

    # calculates the rate of high confidence for a given set of confidences
    @staticmethod
    def calc_conf(confs):
        conf_sum = 0
        for i in confs:
            if i == 1.0:
                conf_sum += 1
        return conf_sum / len(confs)

    # calculates the real rate of high confidence for a given set of trials
    @staticmethod
    def get_real_conf(trials):
        confs = np.empty(len(trials))
        for i, trial in enumerate(trials):
            confs[i] = trial.conf
        return Environment.calc_conf(confs)

    # calculates the predicted rate of high confidence for
    @staticmethod
    def get_pred_conf(trials, cutoff, sigz, sigz_sub, beta, bias=0, mult = 100):
        env = Environment(trials, cutoff)
        agent = Agent(sigz_sub, bias)
        priors = np.zeros((len(trials) * mult, 2))
        values = np.zeros((len(trials) * mult, 2))
        for i, trial in enumerate(trials):
            priors[i * mult : (i + 1) * mult] = np.tile(trial.prior, [mult, 1])
            values[i * mult : (i + 1) * mult] = np.tile(trial.value, [mult, 1])
        obvs = env.obsvs_trials(trials, mult, sigz)

        conf = agent.predict_conf(obvs, priors, values, cutoff, beta)
        return Environment.calc_conf(conf)

    @staticmethod
    def get_pred_conf_other(trials, cutoff, sigz, sigz_sub, conf_type, bias=0, mult = 100):
        env = Environment(trials, cutoff)
        agent = Agent(sigz_sub, bias)
        priors = np.zeros((len(trials) * mult, 2))
        values = np.zeros((len(trials) * mult, 2))
        for i, trial in enumerate(trials):
            priors[i * mult : (i + 1) * mult] = np.tile(trial.prior, [mult, 1])
            values[i * mult : (i + 1) * mult] = np.tile(trial.value, [mult, 1])
        obvs = env.obsvs_trials(trials, mult, sigz)


        if conf_type == 1:  # prob of obsvs
            return Environment.calc_conf(agent.predict_conf_simple(obvs, priors, values, cutoff))
        elif conf_type == 2:  # max post prob
            return Environment.calc_conf(agent.predict_conf_prior(obvs, priors, values, cutoff))
        elif conf_type == 3 :  # max post expected value
            return Environment.calc_conf(agent.predict_conf_value(obvs, priors, values, cutoff))
        else:
            return Environment.calc_conf(agent.predict_conf_entropy(obvs, priors, values, cutoff))
