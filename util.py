from math import sqrt
from matplotlib import pyplot as plt
import time

dist_matrix = {}

# def get_dist_matrix(points_x, points_y):
#     global dist_matrix
#     for i in range(len(points_x)):
#         for j in range(len(points_y)):
#             dist = sqrt((points_x[i] - points_x[j]) ** 2 + (points_y[i] - points_y[j]) ** 2)
#             dist_matrix[(i, j)] = dist
#     return dist_matrix

def get_dist_matrix(points_x, points_y):
    dist_matrix = {}
    for i in range(len(points_x)):
        for j in range(i, len(points_y)):
            dist = sqrt((points_x[i] - points_x[j]) ** 2 + (points_y[i] - points_y[j]) ** 2)
            dist_matrix[(i, j)] = dist
            dist_matrix[(j, i)] = dist
    return dist_matrix

def get_total_dist(feasible_solution, dist_matrix):
    total_dist = 0
    # solution의 경로상 비용을 모두 더한다.
    for i in range(len(feasible_solution)-1):
        total_dist += dist_matrix[(feasible_solution[i], feasible_solution[i+1])]
    # 마지막에 다시 원점으로 돌아오는 거리까지 더해줌
    # total_dist += dist_matrix[(feasible_solution[-1], feasible_solution[0])]

    return total_dist

def show_route(points_x, points_y, solution, generation, val):
    plt.scatter(points_x, points_y)
    plt.plot([points_x[i] for i in solution], [points_y[i] for i in solution], linestyle='-', color='blue',
             label='Line')
    plt.title(f"Generation: {generation},  Total Distance: {val:.2f}")
    plt.show()

