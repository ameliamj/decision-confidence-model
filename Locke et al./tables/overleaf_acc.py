import numpy as np

if __name__ == '__main__':
    precision = 2
    names = ["", "_ee", "_bias", "_bias_combo"]
    titles = ["Normal", "Explore/Exploit", "Bias (symmetric)", "Bias (symmetric \& prior)"]
    table_str = ""
    for i, name in enumerate(names):
        table_str += (titles[i] + " & ")
        aic_accs = np.load(f"../results/acc/vales (pkl)/acc_vales{name}.pkl", allow_pickle=True)

        avg_aic = np.mean(aic_accs, axis=0)
        max_aic = np.max(aic_accs, axis=0)
        min_aic = np.min(aic_accs, axis=0)
        total_aic = np.sum(avg_aic)

        diff = np.maximum(max_aic - avg_aic, avg_aic - min_aic)

        for j in range(5):
            j += 5
            table_str += (str(round(avg_aic[j], precision)) + " $\pm$ " + str(round(diff[j], precision)) + " & ")
        table_str = table_str[:-2]
        table_str += " \\\ \n"

    print(table_str)
