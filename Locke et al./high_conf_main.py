from model.data import Data
from model.environment import Environment
import numpy as np
import pickle

# finds the rate of high confidence for each of the subjects and each of the sessions/directions
if __name__ == '__main__':

    # loads and cleans data
    data = Data('data/rawChoiceData.txt')
    trials_sub = data.split_subject(data.original_trials)
    num_sub = len(trials_sub.keys())
    num_session = 7

    # calculates rate of high confidence overall for each subject
    # and for each session/direction for each subject
    overall_confs = np.zeros(num_sub)
    all_confs = np.zeros((num_sub, num_session, 2))
    for sub in trials_sub:
        overall_confs[int(sub) - 1] = Environment.get_real_conf(trials_sub[sub])
        split_trials = Data.split_session_direction(trials_sub[sub])
        for i in range(num_session):
            right_conf = Environment.get_real_conf(split_trials[f'{i}r'])
            left_conf = Environment.get_real_conf(split_trials[f'{i}l'])
            all_confs[int(sub) - 1, i, :] = right_conf, left_conf

    # writes results
    with open(f'results/other/ high_conf_rates.txt', 'w') as f:
        f.write('overall rates of high confidence for subjects \n')
        f.write(str(overall_confs))
        f.write('\n rates of high confidence for all subjects and sessions/directions \n')
        f.write(str(all_confs))

    with open('results/other/ overall_high_conf.pkl', 'wb') as handle:
        pickle.dump(overall_confs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('overall rates of high confidence for subjects')
    print(overall_confs)
    print('rates of high confidence for all subjects and sessions/directions')
    print(all_confs)

