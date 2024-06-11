import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import read_a_multilist, read_list


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
    PATH = 'results/old_tourn_size_10/no local search/'
    gens_data = load_generation_data(PATH)
    df = prepare_data(gens_data)
    plot_boxplot(df)


def calculate_average_fitness_per_generation(all_runs):
    max_generations = max(len(run) for run in all_runs)
    averages = []

    for i in range(max_generations):
        gen_values = [float(run[i]) for run in all_runs if i < len(run)]
        if gen_values:
            averages.append(sum(gen_values) / len(gen_values))
        else:
            averages.append(None)

    return averages


def plot_fitness(PATH, name):
    elite = [True, False]
    local_search = [False]
    plt.figure(figsize=(10, 6))
    for elite_flag in elite:
        for local_search_flag in local_search:
            fitnesses = read_a_multilist(f'{PATH}{name}/', name, 'fitness_mean_histories',
                                         f'local_search_{local_search_flag}_elite_{elite_flag}')
            # calculate the average fitnesses per generation for every run
            average_fitnesses = calculate_average_fitness_per_generation(fitnesses)
            label = f'Local Search: {local_search_flag}, Elite: {elite_flag}'
            plt.plot(average_fitnesses, label=label)
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Average Mean Fitness population')
    plt.title(f'Average Mean Fitness population per Generation for {name} twodoku')
    plt.show()


def plot_fitness_lines_per_puzzle():
    PATH = 'results/old_tourn_size_10/Experiment with easy/'
    easy_puzzle_names = ['easy_1', 'easy_2', 'easy_3']

    for name in easy_puzzle_names:
        plot_fitness(PATH, name)

if __name__ == "__main__":
    plot_boxplot_generations_per_rates()
    plot_fitness_lines_per_puzzle()