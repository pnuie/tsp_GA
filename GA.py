from test_set import *
from util import *

import random

# test set size = 10, 25, 50 ,75, 200, 500, 1000
class GA:
    def __init__(self, N, population_size, max_generation, mutation_rate):
        self.N = N
        self.points_x, self.points_y = get_test_set(N)
        self.dist_matrix = get_dist_matrix(self.points_x, self.points_y)

        # input
        self.population_size = int(population_size)
        self.max_generation = max_generation
        # self.elitism = elitism
        self.mutation_rate = mutation_rate

        # structure
        self.population = None
        self.parents = None
        self.best_sol = None
        self.best_val = 10e9
        self.generation = 0
        #custom
        self.timeseries = []

    def initialize(self):
        init_sol = []
        for _ in range(self.population_size):
            point_list = list(range(self.N))
            random.shuffle(point_list)
            init_sol.append(point_list)
        init_sol = sorted(init_sol, key = lambda sol: get_total_dist(sol, self.dist_matrix))
        self.population = init_sol

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(self.N), 2))  # 랜덤한 부분 선택
        child = [-1] * self.N
        child[start:end] = parent1[start:end]
        visit = {i:False for i in range(self.N)}
        for i in range(len(child)):
            if not child[i] == -1:
                visit[child[i]] = True
        for i in range(len(child)):
            if child[i] == -1:
                for x in parent2:
                    if visit[x] == True:
                        continue
                    else:
                        visit[x] = True
                        child[i] = x
                        break
                    print(child)
        return child

    def mutation(self, child):
        if random.random() < self.mutation_rate*(1 - self.generation/self.max_generation):
        # if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
        return child

    def elitism(self):
        self.parents = self.population[:int(self.population_size*0.2)]
        self.population = self.population[:int(self.population_size*0.2)]
        # print(self.parents)
        # print(self.population)

    def solve(self):
        st = time.time()
        self.generation = 0
        self.initialize()
        while self.generation < self.max_generation:
            self.generation += 1
            self.elitism()
            while len(self.population) < self.population_size:
                parent1, parent2 = random.sample(self.parents, 2)
                child = self.mutation(self.crossover(parent1, parent2))
                self.population.append(child)
            self.population = sorted(self.population, key=lambda sol: get_total_dist(sol, self.dist_matrix))
            cur_val = get_total_dist(self.population[0], self.dist_matrix)
            self.timeseries.append((self.generation, cur_val))

            if self.best_val >= cur_val:
                self.best_sol = self.population[0]
                self.best_val = cur_val

            if self.generation%100 == 0:
                show_route(self.points_x, self.points_y, self.best_sol, self.generation, self.best_val)

        # def graph(self):

        print(f"elapsed time: {time.time() - st}")
        print(f"best solution: {self.best_sol}")
        print(f"best val: {self.best_val}")
        show_route(self.points_x, self.points_y, self.best_sol,  self.generation, self.best_val)


