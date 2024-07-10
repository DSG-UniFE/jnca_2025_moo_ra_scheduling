import random
import numpy
from copy import copy
import sys

from typing import List, Optional

from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import FloatProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, IntegerSolution, IntegerSolution
from jmetal.operator import IntegerPolynomialMutation
#from jmetal.operator.mutation import Mutation  # Consider to use something more complex like NonUniformMutation or UniformMutation in the next steps
from jmetal.util.archive import CrowdingDistanceArchive, BoundedArchive
from jmetal.util.comparator import DominanceComparator, DominanceWithConstraintsComparator, OverallConstraintViolationComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.density_estimator import CrowdingDistance


"""
Multi-Swarm Particle Swarm Optimization (MSPSO) algorithm for multi-objective problem optimization.
"""

class MSPSO(ParticleSwarmOptimization):
    def __init__(
        self,
        problem: IntegerProblem,
        swarm_size: int,
        termination_criterion: TerminationCriterion, #evaluation of a maximum number of evaluations as default termination criterion
        swarms: dict = {},
        swarm_generator: Generator = store.default_generator, #random generator as default, TODO: more refined generators if any constraints or preferences are given
        swarm_evaluator: Evaluator = store.default_evaluator, #sequential evaluator as default, 'problem.evaluate' method invoked (problem dependent, necessary redefinition if new problem created)
    ):
        super(MSPSO, self).__init__(problem=problem, swarm_size=swarm_size)
        

        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion 
        self.observable.register(termination_criterion)



        #particles speed initialization for generic integer problem
        self.speed = numpy.zeros((self.swarm_size, self.problem.number_of_variables()), dtype=int)

        self.swarms = swarms
        self.global_best = {i: None for i in range(self.problem.number_of_objectives())}
        #repository of non-dominated solutions
        self.non_dominated_archive = CrowdingDistanceArchive(100, dominance_comparator = DominanceComparator())

        #parameters for particles flight
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.r3_min = 0.0
        self.r3_max = 0.1
        self.weight_min = 0.1
        self.weight_max = 0.3
        self.change_velocity1 = -1
        self.change_velocity2 = -1

        #dominance comparator for particles
        self.dominance_comparator = DominanceComparator()

    # Evaluate each swarm
    def evaluate(self, solution_list: List[IntegerSolution]) -> None:
        return self.swarm_evaluator.evaluate(solution_list, self.problem)
    
    
    # Verify if the stopping condition is met
    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met
    
    
    def create_initial_solutions(self) -> List[IntegerSolution]:      
        init_solution = [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size * self.problem.number_of_objectives())]
        return init_solution
    
    
    def create_initial_swarms(self, solutions_list: List[IntegerSolution]) -> dict:
        #Initialize swarms
        self.swarms = {i: [] for i in range(self.problem.number_of_objectives())}
        swarm_index = 0
        #Assign particles to swarms randomly
        for particle in solutions_list:
            if len(self.swarms[swarm_index]) < self.swarm_size:
                self.swarms[swarm_index].append(particle)
            else:
                swarm_index += 1
                self.swarms[swarm_index].append(particle)

        return self.swarms
        
    
    def initialize_velocity(self, swarm: List[IntegerSolution]) -> None:
        for i in range(self.swarm_size):
            for j in range(self.problem.number_of_variables()):
                self.speed[i][j] = 0

    
    # Initialize the local best for each particle
    def initialize_particle_best(self, swarm: List[IntegerSolution]) -> None:
        for particle in swarm:
            particle.attributes["local_best"] = copy(particle)


    # Update the global best of each swarm based on its objective
    def initialize_global_best(self, swarm: List[IntegerSolution], objective_index: int) -> None:
        #best_value = float('inf') if self.problem.obj_directions[objective_index] == self.problem.MINIMIZE else float('-inf')
        best_value = sys.maxsize if self.problem.obj_directions[objective_index] == self.problem.MINIMIZE else -sys.maxsize
        best_particle = None

        for particle in swarm:
            if self.problem.obj_directions[objective_index] == self.problem.MINIMIZE:
                if particle.objectives[objective_index] < best_value:
                    best_value = particle.objectives[objective_index]
                    best_particle = particle
            else:
                if particle.objectives[objective_index] > best_value:
                    best_value = particle.objectives[objective_index]
                    best_particle = particle

            if best_particle is not None:
                self.global_best[objective_index] = copy(best_particle)
    

    def update_archive(self):
        #create temporary archive S
        S = []

        #add every pBest to S
        for swarm in self.swarms.values():
            for particle in swarm:
                S.append(particle.attributes['local_best'])
        #add all current solutions in the archive to S
        for solution in self.non_dominated_archive.solution_list:
            S.append(solution)
        
        self.elitist_learning_strategy(S)
        self.nondominated_solutions_determining(S)


    def update_archive(self):
        #create temporary archive S
        S = []

        #add every pBest to S
        for swarm in self.swarms.values():
            for particle in swarm:
                S.append(particle.attributes['local_best'])
        #add all current solutions in the archive to S
        for solution in self.non_dominated_archive.solution_list:
            S.append(solution)
        
        self.elitist_learning_strategy(S)
        self.nondominated_solutions_determining(S)


    def elitist_learning_strategy(self, new_solutions_set):
        # Perform the ELS on each solution in the archive A
        #perturbation = random.randint(-1, 1)
        mutation_operator = IntegerPolynomialMutation(probability=0.9, distribution_index=20)
        #choose random dimension to perturbate

        for solution in self.non_dominated_archive.solution_list:
            new_solution = self.swarm_generator.new(self.problem)
            new_solution.variables = solution.variables[:]
            # Choose a random dimension to perturbate
            #dimension = random.randint(0, self.problem.number_of_variables() - 1)
            #new_solution.variables[dimension] = new_solution.variables[dimension] + perturbation #(self.problem.upper_bound[dimension] - self.problem.lower_bound[dimension]) * perturbation))
            # Check if the new solution is within the bounds
            #new_solution.variables[dimension] = min(max(new_solution.variables[dimension], self.problem.lower_bound[dimension]), self.problem.upper_bound[dimension])
            # Evaluate the new solution and add it to the new solutions set
            new_solution = mutation_operator.execute(new_solution)
            self.problem.evaluate(new_solution)
            new_solutions_set.append(new_solution)


    def nondominated_solutions_determining(self, new_solutions_set):
        for solution in new_solutions_set:
            self.non_dominated_archive.add(solution)
 

    # Update velocity of particles in each swarm
    def update_velocity(self, swarm: List[IntegerSolution], objective_id: int) -> None:
        global_best = self.global_best[objective_id]
      
        for particle in swarm:
            if self.non_dominated_archive.size() == 0:
                archive_solution = random.choice(self.global_best)
            else:
                archive_solution = random.choice(self.non_dominated_archive.solution_list)
            # from OMOPSO paper: "random variations are a more convenient way of 
            # dealing with the difficulties of fine tuning the parameters 
            # MSPSO paper: The sum of the constriction coefficients is recommended to be 4.0 for best performance 
            r1 = random.uniform(self.r1_min, self.r1_max)
            r2 = random.uniform(self.r2_min, self.r2_max)
            r3 = random.uniform(self.r3_min, self.r3_max)
            c1 = c2 = c3 = 4.0/3 
            w = random.uniform(self.weight_min, self.weight_max)

            for var in range(len(particle.variables)):
                self.speed[swarm.index(particle)][var] = round((
                    w * self.speed[swarm.index(particle)][var]
                    + (c1 * r1 * (particle.attributes["local_best"].variables[var] - particle.variables[var]))
                    + (c2 * r2 * (global_best.variables[var] - particle.variables[var]))
                    + (c3 * r3 * (archive_solution.variables[var] - particle.variables[var]))
                ))


    # Update position of particles in each swarm according to their velocity updates
    def update_position(self, swarm: List[IntegerSolution]) -> None:
        for particle in swarm:
            for i in range(len(particle.variables)):
                particle.variables[i] += round(self.speed[swarm.index(particle)][i])

                # Check if the new position is within the bounds and adjust speed accordingly
                if particle.variables[i] < self.problem.lower_bound[i]:
                    particle.variables[i] = self.problem.lower_bound[i]
                    self.speed[swarm.index(particle)][i] *= self.change_velocity1

                if particle.variables[i] > self.problem.upper_bound[i]:
                    particle.variables[i] = self.problem.upper_bound[i]
                    self.speed[swarm.index(particle)][i] *= self.change_velocity2
                    

    def update_particle_best(self, swarm: List[IntegerSolution], objective_id) -> None:
        for particle in swarm:
            if self.problem.obj_directions[objective_id] == self.problem.MINIMIZE:
                if particle.objectives[objective_id] < particle.attributes['local_best'].objectives[objective_id]:
                    particle.attributes['local_best'] = copy(particle)
            else:
                if particle.objectives[objective_id] > particle.attributes['local_best'].objectives[objective_id]:
                    particle.attributes['local_best'] = copy(particle)

    
    def update_global_best(self, swarm, objective_index) -> None:
        for particle in swarm:
            if self.problem.obj_directions[objective_index] == self.problem.MINIMIZE:
                if particle.attributes['local_best'].objectives[objective_index] < self.global_best[objective_index].objectives[objective_index]:
                    self.global_best[objective_index] = copy(particle.attributes["local_best"])
            else:
                if particle.attributes['local_best'].objectives[objective_index] > self.global_best[objective_index].objectives[objective_index]:
                    self.global_best[objective_index] = copy(particle.attributes["local_best"])


    # Apply random perturbation on swarms leaders
    def perturbation(self, solutions: List[IntegerSolution]) -> None:
        for i in range(self.swarm_size):
            self.random_mutation.execute(solutions[i])


    def init_progress(self) -> None:
        self.evaluations = self.problem.number_of_objectives() * self.swarm_size
        print(f"Initial evaluations: {self.evaluations}")
        self.create_initial_swarms(self.solutions)
        for m in range(self.problem.number_of_objectives()):
            self.initialize_velocity(self.swarms[m])
            self.swarms[m] = self.evaluate(self.swarms[m])
            self.initialize_particle_best(self.swarms[m])
            self.initialize_global_best(self.swarms[m], m)
   
        self.update_archive()

    
    def step(self) -> None:
        for m in range(self.problem.number_of_objectives()):
            self.update_velocity(self.swarms[m], m)
            self.update_position(self.swarms[m])
            self.swarms[m] = self.evaluate(self.swarms[m])
            self.update_particle_best(self.swarms[m], m)
            self.update_global_best(self.swarms[m], m)

        self.update_archive()


    def update_progress(self) -> None:
        self.evaluations += self.swarm_size * self.problem.number_of_objectives()
        #print(f"Evaluations: {self.evaluations}")
        #print(f"Founded {self.non_dominated_archive.size()} non-dominated solutions.")

        observable_data = self.observable_data()
        observable_data["SOLUTIONS"] = self.non_dominated_archive.solution_list
        self.observable.notify_all(**observable_data)


    def get_result(self) -> List[IntegerSolution]:
        return self.non_dominated_archive.solution_list
    
    def result(self) -> List[FloatSolution]:
        return self.get_result()
    
    
    def get_name(self) -> str:
        return "MSPSO"
    
    