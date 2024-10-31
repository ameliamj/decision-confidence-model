from .trial import PriorTrial
import numpy as np


# data cleaning for the data from the priors and payoffs paper specifically
# info about the data is here https://osf.io/e2zrn
# filename: name of the file with the raw data to open
# complete_trials: true if you want data put into CompleteTrial objects, otherwise will
#   default to PriorTrial objects (see trial.py for descriptions of these objects)
class Data:
    def __init__(self, filename='rawChoiceData.txt'):
        # reads in the data
        data = open(filename)
        data = data.readlines()
        data = np.array(data)

        # removes data headings
        data = data[1:]

        # puts data into trials objects and changes representation of left and right
        # left = -1 and right = 1, for stimulus and r1
        trials = []
        for i, trial in enumerate(data):
            trials.append(trial.split()[0].split(','))

            # changing the values for the stimulus and reaction
            if trials[i][5] == '0':  # changes left stimulus from 0 to -1
                trials[i][5] = '-1'
            if trials[i][6] == '0':  # changes left reaction (r1) from 0 to -1
                trials[i][6] = '-1'

            # changing the prior distributions
            if trials[i][2] == '1.00':  # even prior
                prior = [.5, .5]
            elif trials[i][2] == '3.00':  # right biased prior
                prior = [.75, .25]
            else:  # left biased prior
                prior = [.25, .75]

            # changing the value distributions
            if trials[i][3] == '1.00':  # even value
                value = [.5, .5]
            elif trials[i][3] == '2.00':  # right biased value
                value = [2/3, 1/3]
            else:  # left biased value
                value = [1/3, 2/3]

            trials[i] = PriorTrial(float(trials[i][0]), float(trials[i][1]), prior, value, float(trials[i][4]),
                                   float(trials[i][5]), float(trials[i][6]), float(trials[i][7]))

        final_trials = []
        for trial in trials:
            if trial.trial_num > 99:
                final_trials.append(trial)

        self.original_trials = np.array(final_trials)

    # splits trials by subjects
    # returns dictionary of subject number with a list of their trials
    def split_subject(self, all_trials):
        trials = {}
        for trial in all_trials:
            subject_num = trial.subject
            if subject_num not in trials.keys():
                trials[subject_num] = []
            trials[subject_num].append(trial)
        return trials

    # splits trials by their session (each session has diff combo of prior and value distributions)
    # returns a dictionary of the sessions (1-7) and all of their associated trials
    @staticmethod
    def split_session(trials):
        sorted_trials = {}
        for trial in trials:
            if (trial.session - 1) not in sorted_trials.keys():
                sorted_trials[trial.session - 1] = []
            sorted_trials[trial.session - 1].append(trial)
        return sorted_trials

    # splits trials by their direction (whether the stimulus is left or right for the trial)
    # returns a list of right trials and a list of left trials
    @staticmethod
    def split_direction(trials):
        right_trials = []
        left_trials = []
        for trial in trials:
            if trial.stimulus == 1:
                right_trials.append(trial)
            else:
                left_trials.append(trial)
        return right_trials, left_trials

    # splits trials by their session and by direction
    # returns a dictionary with 14 keys for each session and direction combination
    @staticmethod
    def split_session_direction(trials):
        trials_session = Data.split_session(trials)
        trials_split = {}
        for i in trials_session.keys():
            trials_split[str(int(i)) + 'r'], trials_split[str(int(i)) + 'l'] = Data.split_direction(trials_session[i])
        return trials_split