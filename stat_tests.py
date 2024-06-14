import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_list

def load_generation_data(twodoku_name, local_search, elite):
    """ Load generation data based on the provided configurations. """
    return read_list(f'./results/comparison/100_runs/{twodoku_name}/', twodoku_name, 'generation_counts', f'local_search_{local_search}_elite_{elite}')

def perform_levenes_test(*args):
    """ Perform Levene's test for equal variances. """
    stat, p_value = stats.levene(*args)
    print(f"Levene's test: stat = {stat}, p-value = {p_value}")
    return stat, p_value

def perform_anova(*args):
    """ Perform ANOVA test. """
    f_stat, p_value = stats.f_oneway(*args)
    print(f"ANOVA F-statistic: {f_stat}")
    print(f"ANOVA P-value: {p_value}")
    return f_stat, p_value

def perform_t_tests(data1, data2):
    """ Perform t-tests and adjust p-values using Bonferroni correction. """
    t_stat, p_val = stats.ttest_ind(data1, data2)
    corrected_p_val = min(p_val * 2, 1.0)
    return t_stat, p_val, corrected_p_val

def plot_data(data, labels):
    """ Plot data using a boxplot. """
    sns.boxplot(data=data, labels=labels)
    plt.title("Comparison of Generations to Solve Sudoku")
    plt.ylabel("Number of Generations")
    plt.show()

def main():
    twodoku_name = 'easy_3'
    generations_no_local_search = load_generation_data(twodoku_name, False, False)
    generations_with_local_search = load_generation_data(twodoku_name, True, False)
    generations_with_elite_population = load_generation_data(twodoku_name, False, True)

    perform_levenes_test(generations_no_local_search, generations_with_local_search, generations_with_elite_population)
    perform_anova(generations_no_local_search, generations_with_local_search, generations_with_elite_population)

    t_stat1, p_val1, p_val1_corrected = perform_t_tests(generations_no_local_search, generations_with_local_search, generations_with_elite_population)
    t_stat2, p_val2, p_val2_corrected = perform_t_tests(generations_no_local_search, generations_with_local_search, generations_with_elite_population)

    print(f"Comparison No Local Search vs With Local Search: t-statistic = {t_stat1}, p-value = {p_val1}, cor")
    print(f"Comparison No Local Search vs With Elite Population: t-statistic = {t_stat2}, p-value = {p_val2}")

    if p_val1_corrected <= 0.05:
        print("There is a statistically significant difference between No Local Search and With Local Search")
    if p_val2_corrected <= 0.05:
        print("There is a statistically significant difference between No Local Search and With Elite Populat")

    data = [generations_no_local_search, generations_with_local_search, generations_with_elite_population]
    plot_data(data, ["No Local Search", "With Local Search", "With Elite Population"])

if __name__ == "__main__":
    main()
