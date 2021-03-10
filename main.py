from Solution import Solution
from Vessel import Vessel
from Container import Container
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt


filename = "instance1.txt"
input_data = open(filename, 'r')
data = input_data.read().splitlines()
data = [int(line) for line in data]
Vessel = Vessel(data[0], data[1], data[2])
Containers = [Container(c, data[c+3]) for c in range(len(data)-3)]
numberOfContainers = len(Containers)
InitialSolution = Solution(Vessel.nBays, Vessel.nStacks, Vessel.nTiers)
InitialSolution.construct()
InitialSolution.calculateObjective(Containers)
# InitialSolution.print_solution()
print(f"Initial solution: {InitialSolution.objective}")
# Task 1
Improved_Solution = Solution(Vessel.nBays, Vessel.nStacks, Vessel.nTiers)
Improved_Solution.constructionImproved(Containers)
Improved_Solution.calculateObjective(Containers)
Improved_Solution.print_solution()
# This improved structure utilizes domain knowledge/problem spesific knowledge that placing objects near the center of gravity
# Would yield a better solution.
print(f"Construction improved solution: {Improved_Solution.objective}")

# Task 2a
Two_swap_solution = InitialSolution.copy()
Two_swap_solution.local_search_two_swap(Containers)
# Two_swap_solution.print_solution()
print(f"Two swap solution: {Two_swap_solution.objective}")
# Task 2b
Three_swap_solution = InitialSolution.copy()
Three_swap_solution.local_search_three_swap(Containers)
# Three_swap_solution.print_solution()
print(f"Three swap solution: {Three_swap_solution.objective}")
# Task 3A
Tabu_Search_Solution = InitialSolution.copy()
Tabu_Search_Solution.Tabu_Search_Heuristic(Containers, maxIter=100, rand_N=10)

print(f"Tabu Search Solution: {Tabu_Search_Solution.objective}")
# Task 3B
# An aspiration criteria in a Tabu search is an intermediate estimation function to descide if
# a neigbour should be a part of the possible next tabu states. In the Tabu-Search-Implementation in 3A, an aspiration criteria
# could be if the objective value of the neighbour is higher than the initial solution.
