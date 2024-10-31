from model.data import Data
from model.fit import AnalyticalFit, GridFit
from model.likelihood import SdzLike
import random
import numpy as np
from tqdm import tqdm
import time
from model.aic import Aic
from model.param import Param
import pickle

# finds aic accuracy for regular fitting
    # acc_vales.txt
# if you change agent will find aic accuracy for explore/exploit
    # acc_vales_ee.txt
# sigz fit on symm trials and sigz_sub fit on prior trials
if __name__ == '__main__':
    start = time.time()
    reps = 10

    # loads and cleans data
    data = Data('data/rawChoiceData.txt')
    trials_sub = data.split_subject(data.original_trials)
    num_sub = len(trials_sub.keys())

    # keeping track of observational noise and confidence criteria for each subject
    sigzs = np.empty((reps, num_sub))  # external noise
    sigz_subs = np.empty((reps, num_sub))  # internal noise
    aic_accs = np.empty((reps, num_sub))
    run_nums = np.empty(reps)

    # starting values and options
    start_sd = 2.1  # average analytical fit for sd
    symm_iters = 300
    iters = 1000
    alpha = .1
    num_trials_symm = 600
    num_trials = 1200

    for i in range(reps):
        run_num = random.randint(0, 1000)
        run_nums[i] = run_num
        random.seed(run_num)

        for subject in tqdm(trials_sub, desc="fitting for each subject"):
            split_trials = data.split_session(trials_sub[subject])
            symm_trials = split_trials[0]
            prior_trials = np.concatenate((split_trials[1], split_trials[2]))

            random.shuffle(trials_sub[subject])
            random.shuffle(prior_trials)
            random.shuffle(split_trials)

            # sig_z fit analytically on SYMMETRIC trials, sdz fit through grid search on PRIOR trials
            sigz_fit = AnalyticalFit(symm_trials[:num_trials_symm], start_sd, symm_iters, alpha)
            sigz = sigz_fit.fit_sdz()
            sigzs[i, int(subject) - 1] = sigz

            sigz_sub_like = SdzLike()
            sigz_sub_fit = GridFit(prior_trials[:num_trials], sigz, iters, sigz_sub_like,
                                   'sd')  # start from sigz since the ideal observer uses the same value
            sigz_sub = sigz_sub_fit.sdz_fit(sigz)
            sigz_subs[i, int(subject) - 1] = sigz_sub

        num_params = 3

        total_aic_acc, sub_aic_acc = Aic.calc_aic_acc(num_params, sigzs[i], sigz_subs[i], trials_sub)
        aic_accs[i, :] = sub_aic_acc
        print(f'observational noise for each subject: {sigzs[i]}')
        print(f'internal observational noise for each subject: {sigz_subs[i]}')
        print(f'aic for accuracy fit: {total_aic_acc}')
        print(f'aic for accuracy fit by subject: {sub_aic_acc}')

    avg_aic = np.mean(aic_accs, axis=0)
    max_aic = np.max(aic_accs, axis=0)
    min_aic = np.min(aic_accs, axis=0)
    total_aic = np.sum(avg_aic)

    diff = np.maximum(max_aic - avg_aic, avg_aic - min_aic)

    result_str = "["
    for i, aic in enumerate(avg_aic):
        result_str += (str(aic) + " +/- " + str(diff[i]) + " ")
    result_str += "]"
    with open('results/acc/vales (txt)/ acc_vales_ee.txt', 'a') as f:
        f.write(f'total acc aic over {reps} runs \n')
        f.write(str(total_aic))
        f.write(f'\naverage acc aic over {reps} runs \n')
        f.write(result_str)

    param = Param(sigzs, sigz_subs, run_nums)

    with open('results/acc/params/acc_params_ee.pkl', 'wb') as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('results/acc/vales (pkl)/acc_vales_ee.pkl', 'wb') as handle:
        pickle.dump(aic_accs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print(f'total elapsed time: {(end - start) / 60}')




