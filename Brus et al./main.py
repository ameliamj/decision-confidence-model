import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
from agent import Agent
from environment import Environment

# given the data from Brus et al.:
#   - generates confidence predictions for the planning as inference (pai) and expected value (ev) methods
#   - with the real confidence reports, calculates the aic value for each method
def method_comparison(trials):
    real_conf = trials['confidence']
    pred_conf_pai = Agent.pred_conf(trials, conf_type='pai')
    pred_conf_ev = Agent.pred_conf(trials, conf_type='ev')

    aic_pai = Environment.get_aic(real_conf, pred_conf_pai)
    aic_ev = Environment.get_aic(real_conf, pred_conf_ev)

    print(f'aic value for planning as inference: \n {aic_pai}')
    print(f'aic value for expected value: \n {aic_ev}')
    return pred_conf_pai, pred_conf_ev, aic_pai, aic_ev

# given aic values for each subject and for both methods,
# graphs the relationship between the aic values for both methods, where each point represents a subject
# used to generate Fig. 2, middle plot
def graph_aic(subject_aic_pai, subject_aic_ev):
    font = {'weight': 'normal', 'size': 15}
    plt.rc('font', **font)
    plt.xlabel('AIC for soft optimality')
    plt.ylabel('AIC for expected value ratio')
    plt.plot(subject_aic_pai, subject_aic_ev, '.')
    plt.plot(np.arange(0,800,1), 'r--')
    plt.show()

# finds and returns the mean and std of the confidence offset across subjects at the 4 different quantiles
# for the experimental results as well as for the predictions from the pai and ev methods
def find_quantile(correct, corr_pai, corr_ev):
    diff_dict = {'diff_ratings': np.abs(correct['mrUp'] - correct['mrDown'])}
    diff_df = pd.DataFrame(diff_dict)
    result = pd.concat([correct, diff_df], axis=1)
    quartiles = result['diff_ratings'].quantile([.25, .5, .75]).tolist()
    more_correct = {'pai': corr_pai, 'ev': corr_ev, 'quant': result['diff_ratings'].apply(lambda x: 1 if x <= quartiles[0] else (2 if quartiles[0] < x <= quartiles[1] else (3 if quartiles[1] < x <= quartiles[2] else 4)))}
    more_correct = pd.DataFrame(more_correct)

    corr_result = pd.concat([result, more_correct], axis=1)

    reg_means = np.zeros(4)
    pai_means = np.zeros(4)
    ev_means = np.zeros(4)
    reg_stds = np.zeros(4)
    pai_stds = np.zeros(4)
    ev_stds = np.zeros(4)
    for i in range(len(reg_means)):
        reg_means[i] = corr_result.query(f'quant == {i+1}').loc[:, 'confidence'].mean()
        pai_means[i] = corr_result.query(f'quant == {i+1}').loc[:, 'pai'].mean()
        ev_means[i] = corr_result.query(f'quant == {i+1}').loc[:, 'ev'].mean()
        num_subs = corr_result.query(f'quant == {i + 1}').shape[0]
        reg_stds[i] = corr_result.query(f'quant == {i + 1}').loc[:, 'confidence'].std() / np.sqrt(num_subs)
        pai_stds[i] = corr_result.query(f'quant == {i + 1}').loc[:, 'pai'].std() / np.sqrt(num_subs)
        ev_stds[i] = corr_result.query(f'quant == {i + 1}').loc[:, 'ev'].std() / np.sqrt(num_subs)

    return reg_means, pai_means, ev_means, reg_stds, pai_stds, ev_stds

# graphs the mean confidence offset at different quantiles of value difference for experimental results as well as
# confidence predictions from the pai and ev methods
# used to generate Fig. 2, right plot
def quantile_graph(reg_means, pai_means, ev_means, reg_stds, pai_stds, ev_stds):
    quants = [1, 2, 3, 4]

    f = plt.figure()
    plt.figure(figsize=(11, 7))
    font = {'weight': 'normal', 'size': 20}
    plt.rc('font', **font)
    f.set_figwidth(8)
    f.set_figheight(5)
    plt.plot(quants, reg_means - np.min(reg_means), linestyle='-', marker='o', label='experimental', color='tab:blue')
    plt.errorbar(quants, reg_means - np.min(reg_means), yerr=reg_stds, color='tab:blue')
    plt.plot(quants, pai_means - np.min(pai_means), linestyle='-', marker='o', label='soft optimality', color='tab:purple')
    plt.errorbar(quants, pai_means - np.min(pai_means), yerr=pai_stds, color='tab:purple')
    plt.plot(quants, ev_means - np.min(ev_means), linestyle='-', marker='o', label='expected value', color='tab:green')
    plt.errorbar(quants, ev_means - np.min(ev_means), yerr=ev_stds, color='tab:green')
    plt.xticks(quants)
    plt.ylim((-.025, .1))
    plt.legend()
    plt.xlabel('Quantile Value Difference')
    plt.ylabel('Confidence Rating Offset')
    plt.show()

if __name__ == '__main__':
    # load the data
    data = pyreadr.read_r('Data.RData')
    data = data['Data']

    # find the predicted confidences and aic values for each method
    total_conf_pai, total_conf_ev, total_aic_pai, total_aic_ev = method_comparison(data)

    # for each subject, find the predicted confidences and aic values for each method
    subject_data = data.groupby(data.subNr)
    num_subs = int(data.iloc[-1]['subNr'])
    subject_aic_pais = np.zeros(num_subs)
    subject_aic_evs = np.zeros(num_subs)
    for i in range(1, num_subs+1):
        sub_data = subject_data.get_group(i)
        _, _, subject_aic_pais[i - 1], subject_aic_evs[i - 1] = method_comparison(sub_data)

    # for each subject, visualize the relationship between the aic values for different methods (Fig. 2, middle plot)
    graph_aic(subject_aic_pais, subject_aic_evs)

    # find and graph the confidence at each quantile of value differences (Fig. 2, right plot)
    reg_means, pai_means, ev_means, reg_stds, pai_stds, ev_stds = find_quantile(data, total_conf_pai, total_conf_ev)
    quantile_graph(reg_means, pai_means, ev_means, reg_stds, pai_stds, ev_stds)