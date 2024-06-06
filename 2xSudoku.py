import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from collections import Counter
from itertools import combinations, product

from twodokus import easy_twodoku_1, easy_twodoku_2, medium_twodoku_1, solution_easy_twodoku_1, solution_easy_twodoku_2, \
    solution_medium_twodoku_1
from utils import blocks, blocks_to_rows, extract_seq_blocks, fixed_positions, random_fill, save_a_list, \
    save_a_multilist, scores_crossover, \
    single_fitness

random.seed(10)
np.random.seed(10)


class SudokuGA:

    def __init__(self, puzzle,
                 tournament_size=3,
                 population_size=150,
                 mutation_rate=0.5,
                 crossover_rate=0.5,
                 max_generations=100,
                 elite_population_size=50
                 ):
        self.puzzle = blocks(np.array(puzzle))  # make it from normal row-col format to subblock format
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population, self.help_array = self.initialize_population()  # help array is in block format
        self.help_array_rows_format = blocks_to_rows(self.help_array)
        self.fitness_history = []
        self.elite_population = []
        self.elite_population_size = elite_population_size

    def initialize_population(self):
        """
        Randomly fills sudoku in n times , without changing the fixed numbers

        Returns n randomly filled sudokus and the fixed positions of the inital sudoku
        """

        seq = self.puzzle
        fixed = fixed_positions(seq)
        pop = [random_fill(seq) for _ in range(self.population_size)]

        return pop, fixed

    def fitness(self, cand):
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
        Cross over between two parents and
        calculate the scores of
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
        cand = copy.deepcopy(candidate)
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
        def swap_columns(individual):
            # Record all illegal columns in the set C
            C = [c for c in range(6) if len(set(individual[:, c])) != 9]

            if len(C) > 1:
                for illegal_column, other_column in combinations(C, 2):

                    # find the rows that they have repeated numbers
                    repeated_numbers_illegal_column = [(x, index) for index, x in
                                                       enumerate(individual[:, illegal_column])
                                                       if Counter(individual[:, illegal_column])[x] > 1]
                    repeated_numbers_other_column = [(x, index) for index, x in enumerate(individual[:, other_column])
                                                     if Counter(individual[:, other_column])[x] > 1]

                    for value, index in repeated_numbers_illegal_column:
                        for other_value, other_index in repeated_numbers_other_column:
                            # if the repeated numbers are in the same row and cell can be swapped
                            if ((index == other_index) and (self.help_array_rows_format[index, illegal_column] == 0)
                                    and (self.help_array_rows_format[index, other_column] == 0)):
                                if (value not in individual[:, other_column]) and (
                                        other_value not in individual[:, illegal_column]):
                                    # swap the repeated numbers
                                    individual[index, illegal_column], individual[index, other_column] = \
                                        individual[index, other_column], individual[index, illegal_column]
            return individual

        for index, individual in enumerate(self.population):
            temp = blocks_to_rows(individual)
            temp[:9] = swap_columns(temp[:9])
            temp[-9:] = swap_columns(temp[-9:])
            self.population[index] = blocks(temp)

    def row_local_search(self):
        def swap_rows(individual):
            C = [c for c in range(6) if len(set(individual[c, :])) != 9]

            if len(C) > 1:
                for illegal_row, other_row in combinations(C, 2):

                    # find the rows that they have repeated numbers
                    repeated_numbers_illegal_row = [(x, index) for index, x in enumerate(individual[illegal_row, :])
                                                    if Counter(individual[illegal_row, :])[x] > 1]
                    repeated_numbers_other_row = [(x, index) for index, x in enumerate(individual[other_row, :])
                                                  if Counter(individual[other_row, :])[x] > 1]

                    for value, index in repeated_numbers_illegal_row:
                        for other_value, other_index in repeated_numbers_other_row:
                            # if the repeated numbers are in the same row and cell can be swapped
                            if ((index == other_index) and (self.help_array_rows_format[illegal_row, index] == 0)
                                    and (self.help_array_rows_format[other_row, other_index] == 0)):
                                if (value not in individual[other_row, :]) and (
                                        other_value not in individual[illegal_row, :]):
                                    # swap the repeated numbers
                                    individual[illegal_row, index], individual[other_row, index] = \
                                        individual[other_row, index], individual[illegal_row, index]

            return individual

        for index, individual in enumerate(self.population):
            temp = blocks_to_rows(individual)
            temp[:9] = swap_rows(temp[:9])
            temp[-9:] = swap_rows(temp[-9:])
            self.population[index] = blocks(temp)

    def update_elite_population(self):
        # the elite population is a queue structure , that records the best individuals
        # of each generation and updates them with new optimal individuals.

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
        # In elite population learning, the worst individuals in the population are replaced by a random individual
        # xrandom from the elite population or are, or are reinitialized.
        # .We define the probability Pb to control this process.
        worst_individuals = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)[
                            :self.elite_population_size]
        for index,individual in enumerate(worst_individuals):
            random_elite_choice = random.choice(
                self.elite_population)  # TODO check so as not to add the same individual

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

    def solve(self, local_search=True, elite=False):
        for generation in range(self.max_generations):
            # local search
            self.population.sort(key=lambda x: self.fitness(x))

            # Calculate fitness of the best
            best_fitness = self.fitness(self.population[0])

            self.fitness_history.append(best_fitness)
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}")
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

    def plot_fitness_history(self):
        plt.plot(self.fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Best Fitness Value over Generations')
        plt.show()


def run_ga_twodoku(puzzle, solution_puzzle, runs=100, tournament_size=10, population_size=150, mutation_rate=0.3,
                   crossover_rate=0.3, max_generations=5000, local_search=True):
    solution_found = []
    generation_counts = []
    generation_counts_with_sol = []
    times_exec = []
    fitness_histories = []
    for run in range(runs):
        print(f'##### Run = {run}')

        start_time = time.time()

        twodoku = SudokuGA(puzzle=puzzle,
                           tournament_size=tournament_size,
                           population_size=population_size,
                           mutation_rate=mutation_rate,
                           crossover_rate=crossover_rate,
                           max_generations=max_generations)
        solution_pred, generations = twodoku.solve(local_search=local_search)

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
    counter = Counter(solution_found)
    solutions_count_true = counter[True]

    print(
        f'Solutions found in {solutions_count_true} runs and on average in {np.mean(generation_counts_with_sol)} generations.')
    print(f'Average execution time of each {runs} runs in seconds: {np.median(times_exec)}')
    return generation_counts, solution_found, times_exec, fitness_histories


a = [0.2, 0.3, 0.4]
# Generate all combinations of mutation and crossover rates
comb = list(product(a, a))

# Print all combinations
for mut, cross in comb:
    dic = {'Cross': cross, 'Mut': mut}

    print(f"mut_{str(mut)}_cross_{str(cross)}")
    if mut == 0.2 and cross == 0.2:
        generation_counts, solution_found, times_exec, fitness_histories = run_ga_twodoku(medium_twodoku_1,
                                                                                          solution_medium_twodoku_1,
                                                                                          mutation_rate=mut,
                                                                                          crossover_rate=cross,
                                                                                          local_search=True,
                                                                                          population_size=150,
                                                                                          runs=15)

        save_a_list("medium_1", times_exec, "times_exec", f"mut_{str(mut)}_cross_{str(cross)}")
        save_a_list("medium_1", solution_found, "solution_found", f"mut_{str(mut)}_cross_{str(cross)}")
        save_a_list("medium_1", generation_counts, "generation_counts", f"mut_{str(mut)}_cross_{str(cross)}")
        save_a_multilist("medium_1", fitness_histories, "fitness_histories", f"mut_{str(mut)}_cross_{str(cross)}")
