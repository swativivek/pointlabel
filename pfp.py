import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random

class Point:
    def __init__(self, name, x, y):
        self.name = name
        self.posx = x if x != 0 else 0.1
        self.posy = y if y != 0 else 0.1
        self.length = len(name)

class Chromosome:
    def __init__(self, num_points):
        self.gene = np.random.randint(1, 5, size=num_points)  # Initialize genes between 1 and 4
        self.fitness = 0

    def calculate_fitness(self, points, overlap_penalty=2):
        """Optimized fitness function"""
        fitness = 0
        overlap_count = 0
        num_points = len(points)

        for i in range(num_points):
            no_overlap = True
            for j in range(i + 1, num_points):
                overlap = self._check_overlap(points[i], points[j])
                overlap_count += overlap
                if overlap:
                    no_overlap = False
        
            # Reward points that are not overlapping
            if no_overlap:
                fitness += 50  # You can adjust this value
        # Penalize overlaps more heavily
        fitness -= overlap_count * overlap_penalty
        self.fitness = fitness

        return self.fitness


    def _check_overlap(self, point1, point2):
        # Adjust the overlap threshold as needed (e.g., 1.0 may be too small)
        distance = np.sqrt((point1.posx - point2.posx)**2 + (point1.posy - point2.posy)**2)
        return int(distance < 0.5)  # Change this threshold as needed


class Individual:
    def __init__(self, num_points):
        self.chromosome = Chromosome(num_points)
        self.fitness = 0

    def evaluate_fitness(self, points):
        self.fitness = self.chromosome.calculate_fitness(points)
        return self.fitness

class Population:
    def __init__(self, size, num_points):
        self.members = [Individual(num_points) for _ in range(size)]

    def evaluate(self, points):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(ind.evaluate_fitness, points) for ind in self.members]
            for future in futures:
                future.result()

    def select(self):
        # Sort by fitness and select the best individuals
        self.members.sort(key=lambda ind: ind.fitness, reverse=True)
        selected_members = self.members[:len(self.members) // 2]  # Keep top 50%
        print(f"Top fitness in selection: {[ind.fitness for ind in selected_members]}")
        self.members = selected_members


    def reproduce(self):
        # Simple crossover and mutation
        new_members = []
        num_points = len(self.members[0].chromosome.gene)
        
        for _ in range(len(self.members)):
            parent1, parent2 = random.sample(self.members, 2)
            child = Individual(num_points)
            crossover_point = random.randint(0, num_points - 1)
            
            child.chromosome.gene[:crossover_point] = parent1.chromosome.gene[:crossover_point]
            child.chromosome.gene[crossover_point:] = parent2.chromosome.gene[crossover_point:]
            
            # Mutation
            mutation_rate = 0.19
            for i in range(num_points):
                if random.random() < mutation_rate:
                    child.chromosome.gene[i] = random.randint(0, 9)  # Ensure mutation values stay between 1 and 4
            
            new_members.append(child)
        
        self.members = self.members + new_members
        print(f"New members' genes: {[ind.chromosome.gene for ind in new_members]}")

# Main genetic algorithm loop
def genetic_algorithm(num_generations, pop_size, num_points):
    points = [Point(f"Point {i}", random.random(), random.random()) for i in range(num_points)]
    population = Population(pop_size, num_points)

    for generation in range(num_generations):
        print(f"Generation {generation + 1}")
        population.evaluate(points)
        population.select()
        population.reproduce()
        
        # Output the best fitness in this generation
        best_individual = max(population.members, key=lambda ind: ind.fitness)
        print(f"Best fitness: {best_individual.fitness}")

# Run with example parameters
if __name__ == "__main__":
    genetic_algorithm(num_generations=100, pop_size=20, num_points=2)
