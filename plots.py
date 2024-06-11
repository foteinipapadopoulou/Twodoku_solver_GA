import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import read_list


def load_generation_data(PATH):
    mutation_rates = [0.2, 0.3] # TODO
    crossover_rates = [0.2, 0.3] # TODO
    gens_data = {}

    for mut_rate in mutation_rates:
        for cross_rate in crossover_rates:
            key = f'mut_{mut_rate}_cross_{cross_rate}'
            file_name = f'mut_{mut_rate}_cross_{cross_rate}'
            gens_data[key] = read_list(PATH, 'medium_1', 'generation_counts', file_name)

    return gens_data


def prepare_data(gens_data):
    data = []
    for key, values in gens_data.items():
        mutation_rate, crossover_rate = key.split('_')[1], key.split('_')[3]
        for value in values:
            data.append({'Mutation Rate': mutation_rate, 'Crossover Rate': crossover_rate, 'Generations': value})
    return pd.DataFrame(data)


def plot_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Mutation Rate', y='Generations', hue='Crossover Rate', data=df)
    plt.xlabel('Mutation Rate')
    plt.ylabel('Generations')
    plt.title('Generation Counts by Mutation and Crossover Rates')
    plt.legend(title='Crossover Rate')
    plt.show()


def plot_boxplot_generations_per_rates():
    PATH = './results/no local search/'
    gens_data = load_generation_data(PATH)
    df = prepare_data(gens_data)
    plot_boxplot(df)


if __name__ == "__main__":
    plot_boxplot_generations_per_rates()