import numpy as np

if __name__ == '__main__':
    precision = 2
    names = ["", "_combo", "_bias", "_bias_combo"]
    titles = ["Normal", "Symmetric \& Prior", "Bias (symmetric)", "Bias (symmetric \& prior)"]
    models = ["Decision", "Observation", "Posterior", "Expected Value"]
    table_str = ""
    for i, name in enumerate(names):

        aic_confs = np.load(f"../results/conf/vales (pkl)/ conf_vales{name}.pkl", allow_pickle=True)

        for j, model in enumerate(aic_confs):
            table_str += (models[j] + " & ")
            avg_aic = np.mean(model, axis=0)
            max_aic = np.max(model, axis=0)
            min_aic = np.min(model, axis=0)
            total_aic = np.sum(avg_aic)

            diff = np.maximum(max_aic - avg_aic, avg_aic - min_aic)

            for k in range(5):
                k += 5
                table_str += (str(round(avg_aic[k], precision)) + " $\pm$ " + str(round(diff[k], precision)) + " & ")
            table_str = table_str[:-2]
            table_str += " \\\ \n"
        table_str += "\n NEW METHOD \n \n"

    print(table_str)
