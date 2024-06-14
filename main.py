import argparse
from TwodokuGASolver import run_experiment
from twodokus import easy_twodoku_1, easy_twodoku_2, easy_twodoku_3, hard_twodoku_1, hard_twodoku_2, medium_twodoku_1, \
    medium_twodoku_2, \
    solution_easy_twodoku_1, \
    solution_easy_twodoku_2, \
    solution_easy_twodoku_3, solution_hard_twodoku_1, solution_hard_twodoku_2, solution_medium_twodoku_1, \
    solution_medium_twodoku_2


def main():
    parser = argparse.ArgumentParser(description='Run twodoku experiments with configurable parameters.')

    parser.add_argument('--twodoku_level', choices=['easy_1', 'easy_2', 'easy_3', 'medium_1', 'medium_2',
                                                           'hard_1', 'hard_2'],
                               required=True,
                               help='Choose twodoku to run')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs')
    parser.add_argument('--local_search', action='store_true', help='Enable local search')
    parser.add_argument('--elite', action='store_true', help='Enable elite population learning')
    parser.add_argument('--max_gens', type=int, default=10000, help='Maximum number of generations')
    parser.add_argument('--pop_size', type=int, default=150, help='Population size')
    parser.add_argument('--mut', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--cross', type=float, default=0.1, help='Crossover rate')
    parser.add_argument('--path', type=str, required=True, help='Path to save the results')

    args = parser.parse_args()

    twodoku_dict = {
        'easy_1': (easy_twodoku_1, solution_easy_twodoku_1),
        'easy_2': (easy_twodoku_2, solution_easy_twodoku_2),
        'easy_3': (easy_twodoku_3, solution_easy_twodoku_3),
        'medium_1': (medium_twodoku_1, solution_medium_twodoku_1),
        'medium_2': (medium_twodoku_2, solution_medium_twodoku_2),
        'hard_1': (hard_twodoku_1, solution_hard_twodoku_1),
        'hard_2': (hard_twodoku_2, solution_hard_twodoku_2)
    }
    twodoku, solution_twodoku = twodoku_dict[args.twodoku_level]

    print(f"The {args.twodoku_level} twodoku is used.")

    run_experiment(twodoku, solution_twodoku, args.mut, args.cross, args.runs, args.local_search, args.elite,
                       args.max_gens, args.pop_size)


if __name__ == '__main__':
    main()
