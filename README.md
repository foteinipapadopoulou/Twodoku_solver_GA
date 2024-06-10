# Twodoku solver with GA 
This repository contains the source code along with different Twodoku puzzles of 3 different levels of difficulty. The code is written in Python and uses a genetic algorithm to solve the Twodoku puzzles.
Specifically, the file:
* `2xSudoku.py` contains the code for the genetic algorithm.
* `twodokus.py` contains the Twodoku puzzles.
* `utils.py` contains utility functions used in the genetic algorithm.
* `results` folder contains the results of the genetic algorithm for the Twodoku puzzles for local and no local search used.

## How to run the code
1. To run the code, you need to have Python installed on your machine.
2. Clone the repository to your local machine by typing the following command in your terminal:
```git clone https://github.com/foteinipapadopoulou/Twodoku_solver_GA.git```
3. Navigate to the directory where the repository is cloned.
4. Install the required packages by typing the following command in your terminal:
```pip install -r requirements.txt```
5. Open the `2xSudoku.py` file and change the `puzzle` variable or any other variables you want to choose to run for your experiment (e.g. mutation rate etc) in the `run_ga_twodoku`. The puzzles are stored in the `twodokus.py` file.
6. Run the code by typing the following command in your terminal:
```python 2xSudoku.py```
7. The results for each generation will be stored in the `results` folder.

## Example
