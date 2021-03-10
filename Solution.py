import numpy as np
import copy
from Container import sortArrayWeightAscending, sortArrayWeightDescending
from math import floor, ceil, sqrt
import itertools
import random


class Solution:
    def __init__(self, bay, stack, tier):
        self.flowX = np.zeros((bay, stack, tier), dtype=np.int8)
        self.bays = bay
        self.stacks = stack
        self.tiers = tier
        self.objective = float('-inf')
        self.cog = np.array([float('-inf'), float('-inf')])
        self.totalWeightContainers = float('-inf')
        self.hist = []

    def copy(self):
        return copy.deepcopy(self)

    def construct(self):
        i = 0
        for b in range(self.bays):
            for s in range(self.stacks):
                for t in range(self.tiers):
                    self.flowX[b, s, t] = i
                    i += 1

    def constructionImproved(self, Containers):
        # The center goal center of gravity
        gravityGoal = np.array([self.bays/2, self.stacks/2])
        Containers = sortArrayWeightDescending(Containers)

        # Assigning the heaviest containers nearest the center of gravity
        assigned_slots = set()

        while len(Containers) != 0:
            min_tier = (0, 0)
            min_distance = float('Inf')
            for b in range(self.bays):
                for s in range(self.stacks):
                    euclidian_distance = sqrt(
                        (b-0.5+1-gravityGoal[0])**2 + (s-0.5+1-gravityGoal[1])**2)
                    if (b, s) not in assigned_slots and euclidian_distance < min_distance:
                        min_distance = euclidian_distance
                        min_tier = (b, s)

            self.flowX[min_tier[0], min_tier[1]] = [
                container.id for container in Containers[0:self.tiers]]
            assigned_slots.add(min_tier)
            Containers = Containers[self.tiers:]

    def calculateObjective(self, Containers, flowX=None):
        if flowX == None:
            flowX = self.flowX
        gravityGoal = [self.bays/2, self.stacks/2]
        gravityThis = [0, 0]
        sumContainerWeight = 0
        for b in range(self.bays):
            for s in range(self.stacks):
                sumTier = 0
                for t in range(self.tiers):
                    sumTier += Containers[flowX[b, s, t]].weight
                    sumContainerWeight += Containers[flowX[b, s, t]].weight
                gravityThis[0] += (b+1-0.5)*sumTier
                gravityThis[1] += (s+1-0.5)*sumTier
        gravityThis[0] /= sumContainerWeight
        gravityThis[1] /= sumContainerWeight
        evaluation = (gravityGoal[0]-gravityThis[0])**2 + \
            (gravityGoal[1]-gravityThis[1])**2
        self.objective = evaluation
        self.cog = gravityThis
        self.totalWeightContainers = sumContainerWeight

    def local_search_two_swap(self, Containers, has_improved=True, swap_pairs=None):
        if not has_improved:
            return

        bays = self.bays
        tiers = self.tiers
        stacks = self.stacks

        if swap_pairs == None:
            # Create nested list of all positions
            all_positions = np.array([[[(b, s, t) for t in range(tiers)]
                                       for s in range(stacks)] for b in range(bays)])
            all_positions = all_positions.reshape(bays*tiers*stacks, 3)
            # Get all combinations of containers that can be swaped
            swap_pairs = list(itertools.combinations(all_positions, 2))
            # Filter away the combinations where same containers are affected
            # filter_quads = []
        neighbours = []
        # Solve neighbourhood solutions

        for swap_pair in swap_pairs:
            neighbour = self.copy()
            # Perform swap
            neighbour.flowX[tuple(swap_pair[0])], neighbour.flowX[tuple(
                swap_pair[1])] = neighbour.flowX[tuple(swap_pair[1])], neighbour.flowX[tuple(swap_pair[0])]
            neighbour.swap = swap_pair
            neighbour.calculateObjective(Containers)
            neighbours.append(neighbour)

        # Update solution if there is an improvement in the neighbourhood
        has_improved = False
        for neighbour in neighbours:
            if neighbour.objective < self.objective:
                self.flowX = neighbour.flowX
                self.objective = neighbour.objective
                self.cog = neighbour.cog
                self.totalWeightContainers = neighbour.totalWeightContainers
                has_improved = True
                self.last_swap = neighbour.swap
        self.local_search_two_swap(Containers, has_improved, swap_pairs)

    def local_search_three_swap(self, Containers, has_improved=True, swap_triples=None):
        if not has_improved:
            return
        bays = self.bays
        tiers = self.tiers
        stacks = self.stacks
        if swap_triples == None:
            # Create nested list of all positions
            all_positions = np.array([[[(b, s, t) for t in range(tiers)]
                                       for s in range(stacks)] for b in range(bays)])
            all_positions = all_positions.reshape(bays*tiers*stacks, 3)
            # Get all combinations of containers that can be swaped
            swap_triples = tuple(itertools.permutations(all_positions, 3))
        improving_neighbours = []
        for swap_triple in swap_triples:
            neighbour = self.copy()
            # Perform swap
            first = neighbour.flowX[tuple(swap_triple[0])]
            second = neighbour.flowX[tuple(swap_triple[1])]
            third = neighbour.flowX[tuple(swap_triple[2])]

            neighbour.flowX[tuple(swap_triple[0])] = second
            neighbour.flowX[tuple(swap_triple[1])] = third
            neighbour.flowX[tuple(swap_triple[2])] = first
            # Calculate Objective of Neighbour
            neighbour.calculateObjective(Containers)
            if neighbour.objective < self.objective:
                improving_neighbours.append(neighbour)
        has_improved = False
        for neighbour in improving_neighbours:
            if neighbour.objective < self.objective:
                self.flowX = neighbour.flowX
                self.objective = neighbour.objective
                self.cog = neighbour.cog
                self.totalWeightContainers = neighbour.totalWeightContainers
                has_improved = True

        self.local_search_three_swap(Containers, has_improved, swap_triples)

    def Tabu_Search_Heuristic(self, Containers, maxIter=50, tabu_list=(), rand_N=5):
        self.hist.append(self.objective)
        if maxIter == 0:
            return
        bays = self.bays
        tiers = self.tiers
        stacks = self.stacks

        # Create nested list of all positions
        all_positions = np.array([[[(b, s, t) for t in range(tiers)]
                                   for s in range(stacks)] for b in range(bays)])
        all_positions = all_positions.reshape(bays*tiers*stacks, 3)
        # Get all combinations of containers that can be swaped
        swap_pairs = list(itertools.combinations(all_positions, 2))

        possible_swaps = []
        # Only include swaps not in tabulist
        for swap_pair in swap_pairs:
            containers_in_swap = set([tuple(pos) for pos in swap_pair])
            tabu = set(tabu_list)
            if len(tabu.intersection(containers_in_swap)) == 0:
                possible_swaps.append(swap_pair)

        min_neighbour = None
        min_obj = float('Inf')
        # Every rand_N times, choose a random swap
        if maxIter % rand_N == 0:
            swap_pair = random.sample(possible_swaps, 1)[0]
            neighbour = self.copy()
            # Perform swap
            neighbour.flowX[tuple(swap_pair[0])], neighbour.flowX[tuple(
                swap_pair[1])] = neighbour.flowX[tuple(swap_pair[1])], neighbour.flowX[tuple(swap_pair[0])]
            neighbour.swap = swap_pair
            neighbour.calculateObjective(Containers)
            min_neighbour = neighbour
            min_obj = neighbour.objective
        # Calculate all neighbourhoods and find the one with lowest objective value
        else:
            for swap_pair in possible_swaps:
                neighbour = self.copy()
                # Perform swap
                neighbour.flowX[tuple(swap_pair[0])], neighbour.flowX[tuple(
                    swap_pair[1])] = neighbour.flowX[tuple(swap_pair[1])], neighbour.flowX[tuple(swap_pair[0])]
                neighbour.swap = swap_pair
                neighbour.calculateObjective(Containers)
                # Update if the neighbour gives an improvement
                if neighbour.objective < min_obj:
                    min_neighbour = neighbour
                    min_obj = neighbour.objective
        # Update tabulist
        tabu_list += (tuple(min_neighbour.swap[0]),
                      tuple(min_neighbour.swap[1]))
        if len(tabu_list) > 6:
            tabu_list = tabu_list[2:]
        # Update solution
        self.flowX = min_neighbour.flowX
        self.objective = min_neighbour.objective
        self.cog = min_neighbour.cog
        self.totalWeightContainers = min_neighbour.totalWeightContainers
        # self.print_solution()
        self.Tabu_Search_Heuristic(Containers, maxIter-1, tabu_list)

    def print_solution(self):
        print("Current solution:")

        for b in range(self.bays):
            for s in range(self.stacks):
                for t in range(self.tiers):
                    print(f"container: {self.flowX[b,s,t]}")
