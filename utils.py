import copy
import random

import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain

from matplotlib import pyplot as plt


def extract_seq_blocks(puzzle):
    """
    Takes a sudoku puzzle and makes a sequence of every 3x3 block of the puzzle

    Returns a list of sequences, each representing one block
    """
    sequences = []
    puzzle = np.array(puzzle)
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = puzzle[i:i + 3, j:j + 3]  # Extract 3x3 block
            flattened_block = block.flatten()  # Flatten the block into a 1D array
            sequences.append(flattened_block)  # Append the flattened block to sequences
    return sequences


def blocks(puzzle):
    upper_sudoku = extract_seq_blocks(puzzle[:9])
    lower_sudoku = extract_seq_blocks(puzzle[-9:])
    mixed_sudoku = upper_sudoku + lower_sudoku[1:]
    return np.array(mixed_sudoku)


def single_blocks_to_rows(blocks):
    """
    Takes a sequence of 3x3 block of puzzles
    Returns back a sudoku puzzle of row- column format
    """
    rows = [[0] * 9 for _ in range(9)]
    for block_index, block in enumerate(blocks):
        # Determine the starting row and column based on the block index
        start_row = (block_index // 3) * 3
        start_col = (block_index % 3) * 3
        for i in range(9):
            row = start_row + (i // 3)
            col = start_col + (i % 3)
            rows[row][col] = block[i]
    return rows


def blocks_to_rows(cand_blocked):
    upper_sudoku = single_blocks_to_rows(cand_blocked[:9])
    lower_sudoku = single_blocks_to_rows(cand_blocked[-9:])
    return np.array(upper_sudoku + lower_sudoku)


def fixed_positions(seq):
    """
    Takes a sequence, finds the positions that do not have 0 and changes the number to 1
    Returns a sequence with 1s wherever there was a non-zero number in the initial sequence
    """
    seq2 = copy.deepcopy(seq)
    for block_index, block in enumerate(seq):
        for number_index, num in enumerate(block):
            if num != 0:
                seq2[block_index][number_index] = 1
            else:
                seq2[block_index][number_index] = 0
    return seq2


def random_fill(sequences_cand):
    """
    Takes a list of sequences and for each sequence it replaces 0s with a random number between 1 and 9,
    that does not already exist in it.

    Returns the updated list
    """

    sequences = sequences_cand.copy()
    sequence = np.arange(0, 10)
    for seq in sequences:
        new_sequence = [num for num in sequence if num not in seq]
        new_sequence = random.sample(new_sequence, len(new_sequence))
        index = 0
        for i, num in enumerate(seq):
            if num == 0:
                seq[i] = new_sequence[index]
                index += 1
    return sequences


def single_fitness(cand):
    """
    Takes as argument a candidate from the population and
    calculates the repeating integers in each row and each column.

    Returns the sum of the row and column counts
    """
    # Initialize a list to store all rows of the sudoku
    rows_counts = []

    # Iterate over each block and extract elements for each row
    for block_index in range(3):
        for row_index in range(3):
            row = []
            for i in range(3):
                row.extend(cand[block_index * 3 + i][row_index * 3:row_index * 3 + 3])
            rows_counts.append(9 - len(set(row)))

    # Initialize a list to store all columns of the Sudoku
    columns_counts = []

    # Extract columns by iterating over the block sets
    for i in range(3):
        for j in range(3):
            col = []
            for k in range(3):
                col.extend(cand[k * 3 + i][j::3])
            columns_counts.append(9 - len(set(col)))

    return sum(rows_counts + columns_counts)


def single_scores_crossover(cand):
    """
        Takes as argument a candidate from the population and
        calculates the repeating integers in each row and each column.

        Returns the sum of the row and column counts
        """
    # Initialize a list to store all rows of the sudoku
    scores_row = []

    # Iterate over each block and extract elements for each row
    for block_index in range(3):
        unique_row_numbers = []
        for row_index in range(3):
            row = []
            for i in range(3):
                row.extend(cand[block_index * 3 + i][row_index * 3:row_index * 3 + 3])
            unique_row_numbers.append(len(set(row)))
        score_row = sum(unique_row_numbers)
        scores_row.append(score_row)

    # Initialize a list to store all columns of the Sudoku
    unique_column_numbers = []

    # Extract columns by iterating over the block sets
    for i in range(3):
        for j in range(3):
            col = []
            for k in range(3):
                col.extend(cand[k * 3 + i][j::3])
            unique_column_numbers.append(len(set(col)))

    scores_col = []
    for i in range(3):
        score_col = sum(unique_column_numbers[i::3])
        scores_col.append(score_col)
    return scores_row, scores_col


def scores_crossover(cand):
    upper_scores = single_scores_crossover(cand[:9])
    lower_scores = single_scores_crossover(cand[-9:])

    combined_row = upper_scores[0][2] + lower_scores[0][0]
    combined_col = upper_scores[1][2] + lower_scores[1][0]

    scores_row = upper_scores[0][:2] + [combined_row] + lower_scores[0][1:]
    scores_col = upper_scores[1][:2] + [combined_col] + lower_scores[1][1:]
    return scores_row, scores_col


def plot_swarmplot(generations, list_to_plot, name_of_list, x_label):
    # map each values of list to its corresponding generation counts
    list_expanded = list(chain.from_iterable([[rate] * len(gen) for rate, gen in zip(list_to_plot, generations)]))
    generations_expanded = list(chain.from_iterable(generations))

    df_data_swarmplot = pd.DataFrame({
        name_of_list: list_expanded,
        'generations': generations_expanded
    })

    sns.swarmplot(x=name_of_list, y="generations", hue=name_of_list, data=df_data_swarmplot)
    plt.xlabel(x_label)
    plt.ylabel("Generations")
    plt.xticks(range(len(list_expanded)), list_expanded)
    plt.legend()

    plt.show()



def save_a_list(twodoku_name, list_to_save, name_of_list, extra_params):
    np.savetxt(f'{twodoku_name}_{name_of_list}_{extra_params}.txt', list_to_save, delimiter="\n", fmt="%s")


def save_a_multilist(twodoku_name, list_to_save, name_of_list, extra_params):
    with open(f'{twodoku_name}_{name_of_list}_{extra_params}.txt', 'w') as file:
        for item in list_to_save:
            file.write("%s" % item)
            file.write("\n")

# read a multilist
def read_a_multilist(PATH, twodoku_name, name_of_list, extra_params):
    fitnesses_total = []
    with open(f'{PATH}{twodoku_name}_{name_of_list}_{extra_params}.txt', 'r') as file:
        list_to_save = file.readlines()
        for item in list_to_save:
            item = item.replace('[', '').replace(']', '').replace(',', '').split()
            fitnesses = []
            for i in item:
                fitnesses.append(i)
            fitnesses_total.append(fitnesses)
    return fitnesses_total


def is_digit(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_boolean(value):
    return value == 'True' or value == 'False'


def read_list(PATH, twodoku_name, name_of_list, extra_params):
    with open(f'{PATH}{twodoku_name}_{name_of_list}_{extra_params}.txt', 'r') as file:
        list_to_save = file.readlines()
        l = []
        for i in list_to_save:
            i = i.strip()
            if is_digit(i):
                l.append(float(i))
            elif is_boolean(i):
                l.append(i == 'True')
            else:
                l.append(i)

    return l

def calculate_mean_generations_counts(twodoku_name, elite, local_search):
    gens_counts = read_list(f'results/comparison/{twodoku_name}/', twodoku_name, 'generation_counts', f'local_search_{local_search}_elite_{elite}')
    mean_gens = np.mean(gens_counts)
    print(f"Average Generations Count :{mean_gens} for {twodoku_name} and local_search={local_search} and elite={elite}")

def calculate_success_rate(twodoku_name, elite, local_search):
    solutions_found = read_list(f'results/comparison/{twodoku_name}/', twodoku_name, 'solution_found', f'local_search_{local_search}_elite_{elite}')
    success_rate = solutions_found.count(True)
    print(f"Success Rate :{success_rate}/{len(solutions_found)} for {twodoku_name} and local_search={local_search} and elite={elite}")

if __name__ == '__main__':
    twodoku_name = 'easy_1'
    elite = False
    local_search = True
    calculate_mean_generations_counts(twodoku_name, elite, local_search)
    calculate_success_rate(twodoku_name, elite, local_search)
