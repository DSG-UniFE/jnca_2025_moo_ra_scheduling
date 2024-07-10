import numpy as np
import random
import math
import pandas as pd
import os

from jmetal.core.problem import IntegerProblem, FloatProblem
from jmetal.core.solution import IntegerSolution, FloatSolution


class MultiCluster(IntegerProblem):
    """ Multi-Cluster problem """

    def __init__(self):
        super(MultiCluster, self).__init__()
        self.load_requests_data('./cnsm_data/Services.csv')
        self.load_instances_data('./cnsm_data/AWS_EC2_Pricing.csv', './cnsm_data/AWS_EC2_Latency.csv')
        print(f'Number of datacenters: {self.num_datacenters} and number of requests: {self.num_requests}')
        self.num_vars = self.num_datacenters * 3 * 3 * self.num_requests # 3 types of instances, 3 prices (on-demand, reserved, spot)
        print(f'Number of variables: {self.num_vars}')
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.lower_bound = [0 for _ in range(self.num_vars)]
        print(f"Len lower bound: {len(self.lower_bound)}")
        self.upper_bound = [50 for _ in range(self.num_vars)]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
    
    # Load data from files
    def load_requests_data(self,  service_file):
        self.services = pd.read_csv(service_file)
        
        self.requests_cpu = {}
        self.requests_ram = {}
        self.requests_duration = {}
        self.requests_location = {}
        self.replicas = {}

        for idx, row in self.services.iterrows():
            self.requests_cpu[idx] = row['cpu']
            self.requests_ram[idx] = row['ram']
            self.requests_duration[idx] = row['duration']
            self.requests_location[idx] = row['location']
            self.replicas[idx] = row['replicas']

        self.num_requests = len(self.requests_cpu)
    
    def load_instances_data(self, pricing_file, latency_file):
        self.pricing = pd.read_csv(pricing_file)
        self.latency = pd.read_csv(latency_file)

        self.datacenters = self.pricing['DataCenter'].unique()
        self.instances = self.pricing['Instance'].unique()

        #instances cost and capacity data
        self.cost_on_demand = {}
        self.cost_reserved = {}
        self.cost_spot = {}
        self.cpu_capacity = {}
        self.ram_capacity = {}
        self.interrupt_rate = {}

        for idx, row in self.pricing.iterrows():
            dc_instance = (row['DataCenter'], row['Instance'])
            self.cost_on_demand[dc_instance] = row['On-Demand']
            self.cost_reserved[dc_instance] = row['Reserved']
            self.cost_spot[dc_instance] = row['Spot']
            self.cpu_capacity[dc_instance] = row['vCPU']
            self.ram_capacity[dc_instance] = row['RAM']
            self.interrupt_rate[dc_instance] = row['InterruptFrequency'] / 100

        #Latency dictionary
        self.latency_lookup = {}
        for idx, row in self.latency.iterrows():
            from_dc = row['DataCenter']
            for dc in self.datacenters:
                self.latency_lookup[(from_dc, dc)] = row[dc]

        self.num_datacenters = len(self.datacenters)
    
     
    def calculate_costs(self, solution):
        total_costs = {'On-Demand': 0, 'Reserved': 0, 'Spot': 0}
    
        for request_idx in range(self.num_requests):
            for dc_idx, dc in enumerate(self.datacenters):
                for inst_idx, instance in enumerate(self.instances):
                    # Calculate the base index for the current request, datacenter and instance
                    base_index = request_idx * len(self.datacenters) * len(self.instances) * 3 + dc_idx * len(self.instances) * 3 + inst_idx * 3
                    num_replicas_on_demand = solution.variables[base_index]
                    num_replicas_reserved = solution.variables[base_index + 1]
                    num_replicas_spot = solution.variables[base_index + 2]

                    # Get the cost for the current datacenter and instance
                    dc_instance_key = (dc, instance)
                    if dc_instance_key in self.cost_on_demand:
                        total_costs['On-Demand'] += self.cost_on_demand[dc_instance_key] * num_replicas_on_demand * self.requests_duration[request_idx]
                    if dc_instance_key in self.cost_reserved:
                        total_costs['Reserved'] += self.cost_reserved[dc_instance_key] * num_replicas_reserved 
                    if dc_instance_key in self.cost_spot:
                        total_costs['Spot'] += self.cost_spot[dc_instance_key] * num_replicas_spot * self.requests_duration[request_idx]

        # Calculate the total cost by summing the costs for each type of instance
        total_cost = sum(total_costs.values())
        total_costs['total_cost'] = total_cost

        return total_cost
    

    def calculate_max_latency(self, solution):
        max_latency_per_request = [0] * self.num_requests

        num_vars_per_dc = len(self.instances) * 3

        for idx_request in range(self.num_requests):
            request_location = self.requests_location[idx_request]

            for idx_dc, dc in enumerate(self.datacenters):
                # Calculate start and end index in the variables array for this specific request and datacenter
                start_index = idx_request * len(self.datacenters) * num_vars_per_dc + idx_dc * num_vars_per_dc
                end_index = start_index + num_vars_per_dc

                # If at least one replica is deployed in this datacenter, consider the latency
                if any(solution.variables[start_index:end_index]):
                    latency = self.latency_lookup[(request_location, dc)]
                    # Update the maximum latency for this request
                    if latency > max_latency_per_request[idx_request]:
                        max_latency_per_request[idx_request] = max(max_latency_per_request[idx_request], latency)

        return sum(max_latency_per_request)


    def calculate_qos(self, solution):
        total_interruption = 0
        
        for idx_request in range(self.num_requests):
            num_replicas = self.replicas[idx_request]

            for idx_dc, dc in enumerate(self.datacenters):
                for idx_inst, instance in enumerate(self.instances):
                    # Consider Spot types instances only for each datacenter and request (similar to costs, +2 to skip On-Demand and Reserved)
                    spot_index = idx_request * len(self.datacenters) * len(self.instances) * 3 + idx_dc * len(self.instances) * 3 + idx_inst * 3 + 2
                    num_replicas_spot = solution.variables[spot_index] # z variable in the model
                    interruption_rate = self.interrupt_rate[(dc, instance)]

                    total_interruption += (num_replicas_spot * interruption_rate) / num_replicas

        qos = total_interruption / self.num_requests

        return qos

    
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        max_latency = self.calculate_max_latency(solution)
        total_cost = self.calculate_costs(solution)
        qos = self.calculate_qos(solution)
        solution.objectives[0] = max_latency
        solution.objectives[1] = total_cost
        solution.objectives[2] = qos
        solution.constraints[0] = self.check_number_of_replicas(solution)
        solution.constraints[1] = self.check_cpu_capacity(solution)
        solution.constraints[2] = self.check_ram_capacity(solution)
        return solution
       

    def check_number_of_replicas(self, solution):
        replicas_count = [0] * self.num_requests
        replicas_penalty = [0] * self.num_requests
        # required_replicas -allocated replicas <= 0 replicas for each request
        for idx_dc in range(self.num_datacenters):
            for idx_request in range(self.num_requests):
                index = idx_dc * self.num_requests + idx_request
                replicas_count[idx_request] += solution.variables[index]

        for idx_request in range(self.num_requests):
            replicas_penalty[idx_request] = max(0, self.replicas[idx_request] - replicas_count[idx_request])
            #if replicas_count[idx_request] < self.replicas[idx_request]:
            #    return False

        return sum(replicas_penalty)

    # for each request check if the allocated instances provide enough CPU as requested
    # each instance (there are 3 types) has a specific CPU capacity 
    # we need to check if each replica was instantiated on an instance that provides enough CPU
    def check_cpu_capacity(self, solution):
        cpu_constraints = [0] * self.num_requests
        for idx_request in range(self.num_requests):
            request_cpu_penalty = 0
            for idx_dc in range(self.num_datacenters):
                for idx_instance in range(len(self.instances)):
                    index = idx_dc * self.num_requests + idx_request
                    num_replicas = solution.variables[index]
                    if num_replicas > 0:
                        cpu_capacity = self.cpu_capacity[(self.datacenters[idx_dc], self.instances[idx_instance])]
                        cpu_required = self.requests_cpu[idx_request] # CPU required by the request for each replica
                        if cpu_required > cpu_capacity:
                            request_cpu_penalty += cpu_required - cpu_capacity
            cpu_constraints[idx_request] = max(0, request_cpu_penalty)
        return sum(cpu_constraints)
    
    # We do the same for the RAM capacity
    def check_ram_capacity(self, solution):
        ram_constraints = [0] * self.num_requests
        for idx_request in range(self.num_requests):
            request_ram_penalty = 0
            for idx_dc in range(self.num_datacenters):
                for idx_instance in range(len(self.instances)):
                    index = idx_dc * self.num_requests + idx_request
                    num_replicas = solution.variables[index]
                    if num_replicas > 0:
                        ram_capacity = self.ram_capacity[(self.datacenters[idx_dc], self.instances[idx_instance])]
                        ram_required = self.requests_ram[idx_request] # RAM required by the request for each replica
                        if ram_required > ram_capacity:
                            request_ram_penalty += ram_required - ram_capacity
            ram_constraints[idx_request] = max(0, request_ram_penalty)
        return sum(ram_constraints)

    def number_of_constraints(self):
        return 3 #+ self.num_requests


    def name(self):
        return "MultiCluster"
        
   
