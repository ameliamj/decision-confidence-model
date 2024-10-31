import numpy as np
import pickle

from model.data import Data

# finds the right bias for each subject and session which is saved in right bias
if __name__ == '__main__':
    # loads and cleans data
    data = Data('data/rawChoiceData.txt')
    trials_sub = data.split_subject(data.original_trials)
    num_sub = len(trials_sub.keys())
    num_session = 7

    # calculates right bias for each subject overall and for each session for each subject
    total_right_trials = np.zeros(num_sub)
    session_right_trials = np.zeros((num_sub, num_session))
    for sub in trials_sub:
        session_trials = Data.split_session(trials_sub[sub])
        right_total = 0
        for session in session_trials:
            right_session = 0
            for trial in session_trials[session]:
                if trial.reaction == 1:
                    right_total += 1
                    right_session += 1
            right_session /= len(session_trials[session])
            session_right_trials[(int(sub) - 1), int(session)] = right_session
        right_total /= len(trials_sub[sub])
        total_right_trials[int(sub) - 1] = right_total

    # write results
    with open(f'results/other/ right_bias.txt', 'w') as f:
        f.write('total percent of trials that are right by subject \n')
        f.write(str(total_right_trials))
        f.write('\ntotal percent of trials that are right by session and subject \n')
        f.write(str(session_right_trials))

    with open('results/other/ overall_right_bias.pkl', 'wb') as handle:
        pickle.dump(total_right_trials, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('overall right bias for each subject')
    print(total_right_trials)
    print('right bias for each session for all subjects')
    print(session_right_trials)

