import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_list


def load_generation_data(twodoku_name, local_search, elite):
    """ Load generation data based on the provided configurations. """
    return read_list(f'./results/comparison/{twodoku_name}/', twodoku_name, 'generation_counts', f'local_search_{local_search}_elite_{elite}')

def perform_levene_test(data1, data2):
    """ Perform Levene test for homogeneity of variances. """
    stat, p_val = stats.levene(data1, data2)
    return stat, p_val

def perform_t_tests(data1, data2):
    """ Perform t-tests and adjust p-values using Bonferroni correction. """
    t_stat, p_val = stats.ttest_ind(data1, data2)
    corrected_p_val = min(p_val * 2, 1.0)
    return t_stat, p_val, corrected_p_val


def plot_data(data, labels):
    sns.boxplot(data=data)
    plt.title("Comparison of Generations to Solve Sudoku")
    plt.ylabel("Number of Generations")
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.show()

def stat_test_per_puzzle(twodoku_name):
    """ Perform statistical tests for a specific puzzle. """
    generations_baseline = load_generation_data(twodoku_name, False, False)
    generation_local_search = load_generation_data(twodoku_name, True, False)
    generations_elite = load_generation_data(twodoku_name, False, True)
    t_stat1_levene, p_val1_levene = perform_levene_test(generations_baseline, generation_local_search)
    t_stat2_levene, p_val2_levene = perform_levene_test(generations_baseline, generations_elite)

    print(f'Comparison Baseline vs With Local Search: Levene test statistic = {t_stat1_levene:.2f}, p-value = {p_val1_levene:.2f}')
    print(f'Comparison Baseline vs With Elite Population: Levene test statistic = {t_stat2_levene:.2f}, p-value = {p_val2_levene:.2f}')

    t_stat1, p_val1, p_val1_corrected = perform_t_tests(generations_baseline, generation_local_search)
    t_stat2, p_val2, p_val2_corrected = perform_t_tests(generations_baseline, generations_elite)

    print(f"Comparison Baseline vs With Local Search: t-statistic = {t_stat1:.2f}, p-value = {p_val1_corrected:.2f}")
    print(f"Comparison Baseline vs With Elite Population: t-statistic = {t_stat2:.2f}, p-value = {p_val2_corrected:.2f}")

    if p_val1_corrected <= 0.05:
        print("There is a statistically significant difference between Baseline and With Local Search")
    if p_val2_corrected <= 0.05:
        print("There is a statistically significant difference between Baseline and With Elite Population Learning")

    data = [generations_baseline, generation_local_search, generations_elite]
    plot_data(data, ["Baseline", "With Local Search", "With Elite Population"])


if __name__ == "__main__":
    twodoku_names = ['easy_1', 'easy_2', 'easy_3', 'medium_1', 'medium_2', 'hard_1', 'hard_2']
    for twodoku_name in twodoku_names:
        stat_test_per_puzzle(twodoku_name)
