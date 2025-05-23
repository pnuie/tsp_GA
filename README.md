# tsp_GA
This repository explores solutions to the Traveling Salesman Problem (TSP), one of the most well-known NP-hard problems in combinatorial optimization. As part of the learning process for object-oriented programming in Python, custom classes were designed to implement the key components of a Genetic Algorithm (GA), including initialization, selection, crossover, and mutation operators. All operators are coded manually to deepen understanding of both algorithmic design and class-based programming structures.

In addition to the evolutionary approach, this project aims to extend into machine learning-based methods. Specifically, future work will focus on investigating Pointer Networks, a neural network architecture capable of solving combinatorial optimization problems like TSP by learning the structure of optimal tours from data. Through this progression, the repository seeks to bridge classical algorithmic techniques and modern deep learning approaches for combinatorial optimization.

## About Genetic Algorithm
Genetic Algorithm (GA) is a population-based metaheuristic inspired by the process of natural selection. It iteratively evolves a set of candidate solutions toward better solutions by applying genetic operators such as selection, crossover, and mutation. In each generation, individuals with higher fitness are more likely to contribute their genes to the next generation, promoting the exploration and exploitation of the solution space.
GA is widely used for solving complex optimization problems where traditional methods are inefficient or inapplicable, such as the Traveling Salesman Problem (TSP).
<br>
![image](https://github.com/user-attachments/assets/2393db8d-7984-41eb-ae61-322e97e476f7)
<br>

The figures below illustrate how the Genetic Algorithm progressively improves the tour and reduces the total travel distance over generations.

![image](https://github.com/user-attachments/assets/8067c83d-034a-4bb4-868b-7c00c6f50a04)
![image](https://github.com/user-attachments/assets/f60024b3-3b9e-428f-986a-7b1489ea6dc5)
![image](https://github.com/user-attachments/assets/1e6fc8df-c306-46b0-b183-0e3a17ea8792)
![image](https://github.com/user-attachments/assets/3e36567b-3a2c-4471-a319-12d4495fe611)
<br>
The performance of the Genetic Algorithm (GA) is highly sensitive to the choice of hyperparameters. In particular, the mutation rate, population size, and others play critical roles in balancing exploration and exploitation
## Pointer Network
Pointer Networks are a type of neural network architecture designed to solve combinatorial optimization problems where the output is a sequence of discrete elements, such as nodes in a graph.
Unlike traditional models, Pointer Networks generate output sequences by directly "pointing" to elements of the input, making them well-suited for problems like the Traveling Salesman Problem (TSP), where the goal is to find an optimal ordering of cities.
In future work, this project aims to study how Pointer Networks can learn heuristics for TSP and compare their performance to classical algorithms like Genetic Algorithms.
![image](https://github.com/user-attachments/assets/c0aa6eaa-dc46-4ff4-a578-3bbc05c19267)
## Source
https://binaryterms.com/genetic-algorithm-in-data-mining.html
