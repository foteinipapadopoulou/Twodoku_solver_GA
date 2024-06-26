import matplotlib.pyplot as plt
import numpy as np
import random
import time

from collections import Counter
from itertools import combinations, product

from twodokus import easy_twodoku_1, easy_twodoku_2, easy_twodoku_3, hard_twodoku_1, medium_twodoku_1, \
    solution_easy_twodoku_1, \
    solution_easy_twodoku_2, \
    solution_easy_twodoku_3, solution_hard_twodoku_1, solution_medium_twodoku_1
from utils import blocks, blocks_to_rows, extract_seq_blocks, fixed_positions, random_fill, save_a_list, \
    save_a_multilist, scores_crossover, \
    single_fitness

random.seed(10)
np.random.seed(10)


class TwodokuGA:

    def __init__(self, puzzle,
                 tournament_size=3,
                 population_size=150,
                 mutation_rate=0.5,
                 crossover_rate=0.5,
                 max_generations=100,
                 elite_population_size=50
                 ):
        self.puzzle = blocks(np.array(puzzle))  # from normal row-col format to subblock format
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population, self.help_array = self.initialize_population()  # help array is in block format
        self.help_array_rows_format = blocks_to_rows(self.help_array)
        self.fixed_numbers_rows_columns = [(self.puzzle[i][j], i, j) for i in range(9) for j in range(9) if
                                           self.help_array[i][j] == 1]
        self.fitness_history = []
        self.fitness_mean_history = []
        self.fitness_median_history = []

        self.elite_population = []
        self.elite_population_size = elite_population_size

    def initialize_population(self):
        """
        Randomly fills sudoku in n times , without changing the fixed numbers
        Returns n randomly filled puzzles and the fixed positions of the initial twodoku
        """

        seq = self.puzzle
        fixed = fixed_positions(seq)
        pop = [random_fill(seq) for _ in range(self.population_size)]

        return pop, fixed

    def fitness(self, cand):
        """
        Calculate the fitness of a candidate by summing the fitness of the upper and lower sudoku
        """
        upper_fitness = single_fitness(cand[:9])
        lower_fitness = single_fitness(cand[-9:])
        return upper_fitness + lower_fitness

    def tourn_selection(self):
        """
        Samples from the population and returns the one with the best fitness value
        """
        samples = random.sample(self.population, self.tournament_size)
        # sort them and return the best
        samples.sort(key=lambda x: self.fitness(x))
        return samples[0]

    def crossover(self, parent1, parent2):
        """
        Cross over between two parents.
        The crossover is done in two steps:
        1. Row-wise crossover for child1
        2. Column-wise crossover for child2
        """
        if random.random() < self.crossover_rate:
            scores = scores_crossover(parent1)
            scores2 = scores_crossover(parent2)

            indices_row = [(0, 3), (3, 6), (6, 11), (11, 14), (14, 17)]
            child1 = []

            # Row-wise crossover for child1
            for i in range(5):
                if scores[0][i] >= scores2[0][i]:
                    child1.extend(parent1[indices_row[i][0]:indices_row[i][1]])
                else:
                    child1.extend(parent2[indices_row[i][0]:indices_row[i][1]])
            child1 = np.array(child1)

            indices_col = [(0, 3, 6), (1, 4, 7), (2, 5, 8, 11, 14), (9, 12, 15), (10, 13, 16)]
            child2 = np.zeros((17, 9), dtype=int)

            # Column-wise crossover for child2
            for i, (score, score2) in enumerate(zip(scores[1], scores2[1])):
                selected_parent = parent1 if score >= score2 else parent2
                for j in indices_col[i]:
                    child2[j] = selected_parent[j]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, candidate):
        """
        Mutates the candidate by swapping two random numbers in a block
        based on the available numbers that can be changed
        """
        cand = candidate.copy()
        for block_index, subblock in enumerate(cand):
            if random.random() < self.mutation_rate:
                indices = np.arange(9)
                help_array = self.help_array[block_index]

                available_indices = [index for index, flag in zip(indices, help_array) if flag == 0]

                if len(available_indices) >= 2:
                    # just swap two random
                    samples = random.sample(available_indices, 2)
                    value_1 = subblock[samples[0]]
                    value_2 = subblock[samples[1]]
                    cand[block_index][samples[0]] = value_2
                    cand[block_index][samples[1]] = value_1
        return cand

    def column_local_search(self):
        def swap_columns(individual, upper=True):

            if upper:
                # Record all illegal columns in the set C
                C = [c for c in range(6) if len(set(individual[:, c])) != 9]
                help_array = self.help_array_rows_format[:9]
            else:
                C = [c for c in range(6) if
                     len(set(individual[:, c + 3])) != 9]  # Skip the first 3 columns of the lower sudoku
                help_array = self.help_array_rows_format[-9:]

            if len(C) > 1:
                for illegal_column, other_column in combinations(C, 2):

                    # find the blocks that they have repeated numbers
                    repeated_numbers_illegal_column = []
                    for row_index, x in enumerate(individual[:, illegal_column]):
                        if Counter(individual[:, illegal_column])[x] > 1:
                            block_index = illegal_column // 3 + 3 * (row_index // 3)
                            repeated_numbers_illegal_column.append((x, block_index))

                    repeated_numbers_other_column = []
                    for row_index, x in enumerate(individual[:, other_column]):
                        if Counter(individual[:, other_column])[x] > 1:
                            block_other_index = other_column // 3 + 3 * (row_index // 3)
                            repeated_numbers_illegal_column.append((x, block_other_index))

                    for value, index in repeated_numbers_illegal_column:
                        for other_value, other_index in repeated_numbers_other_column:
                            # if the repeated numbers are in the same row and cell can be swapped
                            if ((index == other_index) and (help_array[index, illegal_column] == 0)
                                    and (help_array[index, other_column] == 0)):
                                if (value not in individual[:, other_column]) and (
                                        other_value not in individual[:, illegal_column]):
                                    # swap the repeated numbers
                                    individual[index, illegal_column], individual[index, other_column] = \
                                        individual[index, other_column], individual[index, illegal_column]
            return individual

        for index, individual in enumerate(self.population):
            temp = blocks_to_rows(individual)
            temp[:9] = swap_columns(temp[:9])
            temp[-9:] = swap_columns(temp[-9:], upper=False)
            self.population[index] = blocks(temp)

    def row_local_search(self):
        def swap_rows(individual, upper=True):

            if upper:
                R = [r for r in range(6) if len(set(individual[r, :])) != 9]
                help_array = self.help_array_rows_format[:9]
            else:
                R = [r for r in range(6) if
                     len(set(individual[r + 3, :])) != 9]  # Skip the first 3 rows of the lower sudoku
                help_array = self.help_array_rows_format[-9:]
            if len(R) > 1:
                for illegal_row, other_row in combinations(R, 2):

                    # find the rows that they have repeated numbers
                    repeated_numbers_illegal_row = []
                    for col_index, x in enumerate(individual[illegal_row, :]):
                        if Counter(individual[illegal_row, :])[x] > 1:
                            block_index = col_index // 3 + 3 * (illegal_row // 3)
                            repeated_numbers_illegal_row.append((x, block_index))

                    repeated_numbers_other_row = []
                    for col_index, x in enumerate(individual[other_row, :]):
                        if Counter(individual[other_row, :])[x] > 1:
                            block_other_index = col_index // 3 + 3 * (other_row // 3)
                            repeated_numbers_other_row.append((x, block_other_index))

                    for value, index in repeated_numbers_illegal_row:
                        for other_value, other_index in repeated_numbers_other_row:
                            # if the repeated numbers are in the same row and cell can be swapped
                            if ((index == other_index) and (help_array[illegal_row, index] == 0)
                                    and (help_array[other_row, other_index] == 0)):
                                if (value not in individual[other_row, :]) and (
                                        other_value not in individual[illegal_row, :]):
                                    # swap the repeated numbers
                                    individual[illegal_row, index], individual[other_row, index] = \
                                        individual[other_row, index], individual[illegal_row, index]

            return individual

        for index, individual in enumerate(self.population):
            temp = blocks_to_rows(individual)
            temp[:9] = swap_rows(temp[:9])
            temp[-9:] = swap_rows(temp[-9:], upper=False)
            self.population[index] = blocks(temp)

    def update_elite_population(self):
        """
        the elite population is a queue structure , that records the best individuals
        of each generation and updates them with new optimal individuals.
        """

        for individual in self.population:
            fitness = self.fitness(individual)
            if len(self.elite_population) < self.elite_population_size:
                self.elite_population.append(individual.copy())

                # sort the elite population based on the fitness value
                sorted(self.elite_population, key=lambda x: self.fitness(x))

            elif fitness < self.fitness(self.elite_population[-1]):
                self.elite_population[-1] = individual.copy()
                # sort the elite population based on the fitness value
                sorted(self.elite_population, key=lambda x: self.fitness(x))

    def elite_population_learning(self):
        """
        In elite population learning, the worst individuals in the population
        are replaced by a random individual xrandom from the elite population
        or are, or are reinitialized based on a probability Pb to control this process.
        """
        worst_individuals = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)[
                            :self.elite_population_size]
        for index, individual in enumerate(worst_individuals):
            random_elite_choice = random.choice(
                self.elite_population)

            pb = (self.fitness(individual) - self.fitness(random_elite_choice)) / self.fitness(individual)
            if random.random() < pb:
                # Replace with random elite individual
                worst_individuals[index] = random_elite_choice
            else:  # Reinitialize
                seq = self.puzzle
                worst_individuals[index] = random_fill(seq)

    def evolve(self):
        new_population = []

        # loop until fill the population list
        while len(new_population) < self.population_size:
            parent1 = self.tourn_selection()
            parent2 = self.tourn_selection()
            # cross over
            child1, child2 = self.crossover(np.array(parent1), np.array(parent2))
            # mutation
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        self.population = new_population[:self.population_size]

    def check_populations(self):
        for individual in self.population:
            if self.check_valid_pop(individual) is False:
                print("Invalid Population")
                return False
        return True

    def check_valid_pop(self, child):
        for number, row, column in self.fixed_numbers_rows_columns:
            if child[row][column] != number:
                print(f"Change is in {number}")
                print("Removing wrong pop")
                return False
        return True

    def solve(self, local_search=True, elite=False):
        for generation in range(self.max_generations):
            # local search
            self.population.sort(key=lambda x: self.fitness(x))

            # Calculate fitness of the best
            best_fitness = self.fitness(self.population[0])
            pop_fitnesses = [self.fitness(individual) for individual in self.population]
            # Calculate mean and median fitness of the population
            mean_fitness = np.mean(pop_fitnesses)
            median_fitness = np.median(pop_fitnesses)

            self.fitness_history.append(best_fitness)
            self.fitness_mean_history.append(mean_fitness)
            self.fitness_median_history.append(median_fitness)

            if generation % 100 == 0:
                print(
                    f"Generation {generation}: Best fitness = {best_fitness}, Mean fitness: = {mean_fitness}, Median fitness: = {median_fitness}")
            if best_fitness == 0:
                # Found Solution
                print(f"Found solution in Generation {generation}")
                return self.population[0], generation

            # Selection , Mutation and Cross over
            self.evolve()
            if local_search:
                self.column_local_search()
                self.row_local_search()

            if elite:
                # Update the elite population
                self.update_elite_population()

                # Elite population learning
                self.elite_population_learning()

        return None, self.max_generations


def run_ga_twodoku(puzzle, solution_puzzle, runs=100, tournament_size=3, population_size=150, mutation_rate=0.3,
                   crossover_rate=0.3, max_generations=5000, local_search=True, elite=False):
    solution_found = []
    generation_counts = []
    generation_counts_with_sol = []
    times_exec = []
    fitness_histories = []
    fitness_mean_histories = []
    fitness_median_histories = []

    for run in range(runs):
        print(f'##### Run = {run}')

        start_time = time.time()

        twodoku = TwodokuGA(puzzle=puzzle,
                            tournament_size=tournament_size,
                            population_size=population_size,
                            mutation_rate=mutation_rate,
                            crossover_rate=crossover_rate,
                            max_generations=max_generations)
        solution_pred, generations = twodoku.solve(local_search=local_search, elite=elite)

        total_time = time.time() - start_time
        times_exec.append(total_time)

        if solution_pred is None:
            solution_found.append(False)
        else:
            print(blocks_to_rows(solution_pred))
            np.testing.assert_array_equal(blocks_to_rows(solution_pred), np.array(solution_puzzle))
            solution_found.append(True)
            generation_counts_with_sol.append(generations)

        generation_counts.append(generations)
        fitness_histories.append(twodoku.fitness_history)
        fitness_mean_histories.append(twodoku.fitness_mean_history)
        fitness_median_histories.append(twodoku.fitness_median_history)

    counter = Counter(solution_found)
    solutions_count_true = counter[True]

    print(
        f'Solutions found in {solutions_count_true} runs and on average in {np.mean(generation_counts_with_sol)} generations.')
    print(f'Average execution time of each {runs} runs in seconds: {np.median(times_exec)}')
    return generation_counts, solution_found, fitness_histories, fitness_mean_histories, fitness_median_histories


def run_ga_mutation_crossover_rates(twodoku, solution_twodoku, runs, local_search, elite, max_gens, pop_size=150):
    cross_rates = [0.1, 0.2, 0.3, 0.4]
    mut_rates = [0.05, 0.1, 0.15, 0.2]
    # Generate all combinations of mutation and crossover rates
    comb = list(product(mut_rates, cross_rates))

    # Print all combinations
    for mut, cross in comb:
        print(f'Running experiment:\n---------------\n'
              f'runs = {runs}\n'
              f'mutation rate = {mut}\n'
              f'crossover rate = {cross}\n'
              f'local search = {local_search}\n'
              f'elite = {elite}\n'
              f'max generations = {max_gens}\n'
              f'population size = {pop_size}\n---------------\n')

        generation_counts, solution_found, fitness_histories, fitness_mean_histories, fitness_median_histories = run_ga_twodoku(
            twodoku,
            solution_twodoku,
            mutation_rate=mut,
            crossover_rate=cross,
            local_search=local_search,
            elite=elite,
            population_size=pop_size,
            max_generations=max_gens,
            runs=runs)
        extra_params = f'mut_{str(mut)}_cross_{str(cross)}'
        save_a_list(f'{PATH}{twodoku_name}', solution_found, "solution_found", extra_params)
        save_a_list(f'{PATH}{twodoku_name}', generation_counts, "generation_counts", extra_params)
        save_a_multilist(f'{PATH}{twodoku_name}', fitness_histories, "fitness_histories", extra_params)
        save_a_multilist(f'{PATH}{twodoku_name}', fitness_mean_histories, "fitness_mean_histories",
                         extra_params)
        save_a_multilist(f'{PATH}{twodoku_name}', fitness_median_histories, "fitness_median_histories",
                             extra_params)
        print("Experiment finished")


def run_experiment(twodoku, solution_twodoku, mut, cross, runs, local_search, elite, max_gens, pop_size=150):
    print(f'Running experiment:\n---------------\n'
          f'runs = {runs}\n'
          f'mutation rate = {mut}\n'
          f'crossover rate = {cross}\n'
          f'local search = {local_search}\n'
          f'elite = {elite}\n'
          f'max generations = {max_gens}\n'
          f'population size = {pop_size}\n---------------\n')
    generation_counts, solution_found, fitness_histories, fitness_mean_histories, fitness_median_histories = run_ga_twodoku(
        twodoku, solution_twodoku,
        mutation_rate=mut,
        crossover_rate=cross,
        local_search=local_search,
        elite=elite,
        population_size=pop_size,
        runs=runs,
        max_generations=max_gens)
    extra_params = f"local_search_{str(local_search)}_elite_{str(elite)}"
    save_a_list(f'{PATH}{twodoku_name}', solution_found, "solution_found", extra_params)
    save_a_list(f'{PATH}{twodoku_name}', generation_counts, "generation_counts", extra_params)
    save_a_multilist(f'{PATH}{twodoku_name}', fitness_histories, "fitness_histories", extra_params)
    save_a_multilist(f'{PATH}{twodoku_name}', fitness_mean_histories, "fitness_mean_histories",
                     extra_params)
    save_a_multilist(f'{PATH}{twodoku_name}', fitness_median_histories, "fitness_median_histories",
                     extra_params)
    print("Experiment finished")


if __name__ == '__main__':
    # Change this to TRUE to run the rates experiment
    RUN_RATES_EXPERIMENT = False

    if RUN_RATES_EXPERIMENT is True:
        PATH = './results/rates/'
        twodoku_name = 'easy_1'
        twodoku = easy_twodoku_1
        solution_twodoku = solution_easy_twodoku_1
        print(f"The {twodoku_name} twodoku is used.")
        runs = 15
        local_search = False
        elite = False
        max_gens = 5000
        pop_size = 150
        run_ga_mutation_crossover_rates(twodoku, solution_twodoku, runs=runs, local_search=local_search, elite=elite,
                                        max_gens=max_gens, pop_size=pop_size)

    RUN_EXPERIMENT = True
    if RUN_EXPERIMENT is True:
        PATH = 'results/old_trials/100_runs/easy_3/'
        twodoku_name = 'easy_3'
        twodoku = easy_twodoku_3
        solution_twodoku = solution_easy_twodoku_3
        print(f"The {twodoku_name} twodoku is used.")
        mut = 0.1
        cross = 0.1
        runs = 100
        local_search = False
        elite = True
        max_gens = 10000
        pop_size = 150
        run_experiment(twodoku=twodoku, solution_twodoku=solution_twodoku, mut=mut, cross=cross, runs=runs,
                       local_search=local_search, elite=elite, max_gens=max_gens, pop_size=pop_size)
