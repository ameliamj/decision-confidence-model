from model.data import Data
from model.fit import AnalyticalFit, GridFit
from model.likelihood import  SdzLike, BiasLike
import random
import numpy as np
from tqdm import tqdm
import time
from model.aic import Aic
from model.param import BiasParam
import pickle

# finds aic accuracy when bias is fit to symm trials
    # acc_vales_bias.txt
# and when bias is fit to a mixture of symm and prior trials
    # acc_vales_bias_combo.txt
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
    sigz_subs = np.empty((reps, num_sub)) # internal noise
    biases = np.empty((reps, num_sub))
    biases_combo = np.empty((reps, num_sub))
    aic_accs_bias = np.zeros((reps, num_sub))
    aic_accs_bias_combo = np.zeros((reps, num_sub))
    run_nums = np.empty(reps)

    # starting values and options
    start_sd = 2.1  # average analytical fit for sd
    start_bias = 0
    symm_iters = 300
    iters = 1000
    alpha = .1
    beta = 1
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
            symm_prior = np.concatenate((symm_trials, prior_trials)) # fitting conf on symm and prior right now

            random.shuffle(trials_sub[subject])
            random.shuffle(prior_trials)
            random.shuffle(split_trials)
            random.shuffle(symm_prior)

            # sig_z fit analytically on SYMMETRIC trials, sig_z_sub fit through grid search on PRIOR trials
            sigz_fit = AnalyticalFit(symm_trials[:num_trials_symm], start_sd, symm_iters, alpha)
            sigz = sigz_fit.fit_sdz()
            sigzs[i, int(subject) - 1] = sigz

            sigz_sub_like = SdzLike()
            sigz_sub_fit = GridFit(prior_trials[:num_trials], sigz, iters, sigz_sub_like,
                                   'sd')  # start from sigz since the ideal observer uses the same value
            sigz_sub = sigz_sub_fit.sdz_fit(sigz)
            sigz_subs[i, int(subject) - 1] = sigz_sub

            bias_like = BiasLike()
            bias_fit = GridFit(symm_trials[:num_trials_symm], start_bias, iters, bias_like, 'bias')
            bias = bias_fit.bias_fit(sigz, sigz_sub)
            biases[i, int(subject) - 1] = bias

            bias_combo_fit = GridFit(symm_prior[:num_trials_symm], start_bias, iters, bias_like, 'bias')
            bias_combo = bias_combo_fit.bias_fit(sigz, sigz_sub)
            biases_combo[i, int(subject) - 1] = bias_combo


        num_params = 3
        avg_bias, sub_bias = Aic.calc_aic_acc_bias(num_params, sigzs[i], sigz_subs[i], biases[i], trials_sub)
        aic_accs_bias[i, :] = sub_bias
        avg_bias_combo, sub_bias_combo = Aic.calc_aic_acc_bias(num_params, sigzs[i], sigz_subs[i], biases_combo[i], trials_sub)
        aic_accs_bias_combo[i, :] = sub_bias_combo
        print(f'observational noise for each subject: {sigzs[i]}')
        print(f'internal observational noise for each subject: {sigz_subs[i]}')
        print(f'bias terms for each subject: {biases[i]}')
        print(f'total aic for accuracy fit w/ bias: {avg_bias}')
        print(f'by subject aic for accuracy fit w/ bias: {sub_bias}')

    avg_aic = np.mean(aic_accs_bias, axis=0)
    max_aic = np.max(aic_accs_bias, axis=0)
    min_aic = np.min(aic_accs_bias, axis=0)

    avg_aic_combo = np.mean(aic_accs_bias_combo, axis=0)
    max_aic_combo = np.max(aic_accs_bias_combo, axis=0)
    min_aic_combo = np.min(aic_accs_bias_combo, axis=0)

    diff = np.maximum((max_aic - avg_aic), (avg_aic-min_aic))
    diff_combo = np.maximum((max_aic_combo - avg_aic_combo), (avg_aic_combo - min_aic_combo))

    result_str = "["
    for i, aic in enumerate(avg_aic):
        result_str += (str(aic) + " +/- " + str(diff[i]) + " ")
    result_str += "]"
    with open('results/acc/vales (txt)/acc_vales_bias.txt', 'a') as f:
        f.write(f'total acc aic over {reps} runs \n')
        f.write(str(np.sum(avg_aic)))
        f.write(f'\naverage acc aic over {reps} runs \n')
        f.write(result_str)

    param = BiasParam(sigzs, sigz_subs, run_nums, biases)

    with open('results/acc/params/acc_params_bias.pkl', 'wb') as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('results/acc/vales (pkl)/acc_vales_bias.pkl', 'wb') as handle:
        pickle.dump(aic_accs_bias, handle, protocol=pickle.HIGHEST_PROTOCOL)

    result_str = "["
    for i, aic in enumerate(avg_aic_combo):
        result_str += (str(aic) + " +/- " + str(diff_combo[i]) + " ")
    result_str += "]"
    with open('results/acc/vales (txt)/acc_vales_bias_combo.txt', 'a') as f:
        f.write(f'total acc aic over {reps} runs \n')
        f.write(str(np.sum(avg_aic_combo)))
        f.write(f'\naverage acc aic over {reps} runs \n')
        f.write(result_str)

    param = BiasParam(sigzs, sigz_subs, run_nums, biases_combo)

    with open('results/acc/params/acc_params_bias_combo.pkl', 'wb') as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('results/acc/vales (pkl)/acc_vales_bias_combo.pkl', 'wb') as handle:
        pickle.dump(aic_accs_bias_combo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print(f'total elapsed time: {(end - start) / 60}')




