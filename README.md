# Twodoku solver with GA 
This repository contains the source code along with different Twodoku puzzles of 3 different levels of difficulty. The code is written in Python and uses a genetic algorithm to solve the Twodoku puzzles.
Specifically, the file:
* `TwodokuGASolver.py` contains the code for the genetic algorithm.
* `main.py` contains the code to run the genetic algorithm using arguments.
* `twodokus.py` contains the Twodoku puzzles.
* `utils.py` contains utility functions used in the genetic algorithm.
* `results` folder contains the results of the genetic algorithm for the Twodoku puzzles for different experiments
* `stat_tests.py` contains the code to perform the statistical tests.
* `plots.py` contains the code to generate the plots given in the report.

## Installation

To run the code, you need to have Python 3.x installed on your machine.

1. Clone the repository to your local machine by typing the following command in your terminal:
```git clone https://github.com/foteinipapadopoulou/Twodoku_solver_GA.git```
3. Navigate to the directory where the repository is cloned.
4. Install the required packages by typing the following command in your terminal:
```pip install -r requirements.txt```

## Usage
1. Run the `main.py` script by using the command line with the following options:
```python script_name.py --twodoku <twodoku_level> [options]```
   1. Required Arguments
        
      `--twodoku`: Choose the Twodoku level to run. Choices are: `easy_1`, `easy_2`, `easy_3`, `medium_1`, `medium_2`, `hard_1`, `hard_2`.
      
      `--path`: Path to save the results.
   2. Optional Arguments
   
       `--runs`: Number of runs (default is 100).

       `--local_search`: Enable local search (flag, default is False).
      
       `--elite`: Enable elite population learning (flag, default is False).
       
       `--max_gens`: Maximum number of generations (default is 10000).
       
       `--pop_size`: Population size (default is 150).
       
       `--mut`: Mutation rate (default is 0.1).
       
       `--cross`: Crossover rate (default is 0.1).

2. The results will be stored in the specified `path` folder.

## Example
To run the genetic algorithm for the Twodoku puzzle `easy_1` with local search enabled, mutation rate 0.1 and save the results in the `results` folder, type the following command in your terminal:

```python main.py --twodoku_level easy_1 --path ./results/ --local_search --mut 0.1```

Then an experiment will be run with the specified parameters and the results will be saved in the `results` folder. The console will print the selected parameters and for every run and per 100 generations will print the best fitness value, the mean and the median fitness value of the population.
```
The easy_3 twodoku is used.
Running experiment:
---------------
runs = 100
mutation rate = 0.1
crossover rate = 0.1
local search = True
elite = False
max generations = 10000
population size = 150
---------------

##### Run = 0
Generation 0: Best fitness = 37, Mean fitness: = 50.56666666666667, Median fitness: = 50.0
Found solution in Generation 31
[[3 9 7 1 5 6 2 4 8]
 [2 8 6 3 9 4 1 7 5]
 [5 4 1 2 8 7 3 9 6]
 [1 3 5 7 2 8 9 6 4]
 [4 6 8 9 3 1 7 5 2]
 [7 2 9 6 4 5 8 1 3]
 [8 1 3 5 6 9 4 2 7]
 [6 7 2 4 1 3 5 8 9]
 [9 5 4 8 7 2 6 3 1]
 [4 2 7 1 3 5 6 9 8]
 [5 8 9 4 2 6 3 1 7]
 [6 3 1 8 9 7 5 2 4]
 [1 6 4 9 7 2 8 3 5]
 [8 9 2 5 4 3 1 7 6]
 [3 7 5 6 8 1 2 4 9]
 [2 1 8 7 5 4 9 6 3]
 [9 4 6 3 1 8 7 5 2]
 [7 5 3 2 6 9 4 8 1]]
 (After the specified number of runs).........
Solutions found in 100 runs and on average in 41.26 generations.
Average execution time of each 100 runs in seconds: 16.42952847480774
Experiment finished
```
