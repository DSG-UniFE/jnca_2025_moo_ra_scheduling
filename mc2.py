import numpy as np
import random
import math
import pandas as pd
import os

from jmetal.core.problem import IntegerProblem, FloatProblem
from jmetal.core.solution import IntegerSolution, FloatSolution

class MC2(IntegerProblem):
    """ Multi-Cluster problem """

    def __init__(self):
        super(MC2, self).__init__()
        self.load_requests_data('./cnsm_data/Services.csv')
        self.load_instances_data('./cnsm_data/AWS_EC2_Pricing.csv', './cnsm_data/AWS_EC2_Latency.csv')
        print(f'Number of datacenters: {self.num_datacenters} and number of requests: {self.num_requests}')
        
        # Calcola il numero totale di repliche
        self.num_replicas = sum(self.replicas.values())
        self.mapping = self.create_replica_mapping()
        
        self.num_combinations = self.num_datacenters * 3 * 3  # 3 tipi di istanza, 3 prezzi (on-demand, reserved, spot)
        
        print(f'Number of replicas: {self.num_replicas}')
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        
        self.lower_bound = [0 for _ in range(self.num_replicas)]
        self.upper_bound = [self.num_combinations - 1 for _ in range(self.num_replicas)]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
    
    def create_replica_mapping(self):
        mapping = []
        for request_idx, num_replicas in self.replicas.items():
            for replica_idx in range(num_replicas):
                mapping.append(request_idx)
        return mapping
    
    # Load data from files
    def load_requests_data(self, service_file):
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

        # Instances cost and capacity data
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

        # Latency dictionary
        self.latency_lookup = {}
        for idx, row in self.latency.iterrows():
            from_dc = row['DataCenter']
            for dc in self.datacenters:
                self.latency_lookup[(from_dc, dc)] = row[dc]

        self.num_datacenters = len(self.datacenters)
    
    def encode(self, dc_idx, instance_idx, price_idx):
        return dc_idx * 9 + instance_idx * 3 + price_idx

    def decode(self, value):
        dc_idx = value // 9
        instance_idx = (value % 9) // 3
        price_idx = value % 3
        return dc_idx, instance_idx, price_idx
     
    def calculate_costs(self, solution):
        total_costs = {'On-Demand': 0, 'Reserved': 0, 'Spot': 0}
        instance_usage = {}  # Traccia l'uso delle istanze
        cpu_violations = 0
        ram_violations = 0

        for idx, value in enumerate(solution.variables):
            dc_idx, instance_idx, price_idx = self.decode(value)
            dc = self.datacenters[dc_idx]
            instance = self.instances[instance_idx]
            request_idx = self.mapping[idx]
            cpu_required = self.requests_cpu[request_idx]
            ram_required = self.requests_ram[request_idx]

            dc_instance_key = (dc, instance, price_idx)
            if dc_instance_key not in instance_usage:
                # cpu and ram are resetted to 0 each time a new instance/bin is opened
                instance_usage[dc_instance_key] = {'cpu': 0, 'ram': 0, 'count': 0}

            cpu_capacity = self.cpu_capacity[(dc, instance)]
            ram_capacity = self.ram_capacity[(dc, instance)]

            if (instance_usage[dc_instance_key]['cpu'] + cpu_required <= cpu_capacity) and \
            (instance_usage[dc_instance_key]['ram'] + ram_required <= ram_capacity):
                # Allocate the given instance in the same bin
                instance_usage[dc_instance_key]['cpu'] += cpu_required
                instance_usage[dc_instance_key]['ram'] += ram_required
            else:
                # Allocate the instance to a different instance
                # And then add the cost
                instance_usage[dc_instance_key]['cpu'] = cpu_required
                instance_usage[dc_instance_key]['ram'] = ram_required
                instance_usage[dc_instance_key]['count'] += 1
                if price_idx == 0:  # On-Demand
                    total_costs['On-Demand'] += self.cost_on_demand[(dc, instance)] * self.requests_duration[request_idx]
                elif price_idx == 1:  # Reserved
                    total_costs['Reserved'] += self.cost_reserved[(dc, instance)]
                elif price_idx == 2:  # Spot
                    total_costs['Spot'] += self.cost_spot[(dc, instance)] * self.requests_duration[request_idx]

            # Check CPU and RAM violations
            if instance_usage[dc_instance_key]['cpu'] > cpu_capacity:
                cpu_violations += max(0, instance_usage[dc_instance_key]['cpu'] - cpu_capacity)
            if instance_usage[dc_instance_key]['ram'] > ram_capacity:
                ram_violations += max(0, instance_usage[dc_instance_key]['ram'] - ram_capacity)

        total_cost = sum(total_costs.values())
        total_costs['total_cost'] = total_cost

        return total_cost, cpu_violations, ram_violations, instance_usage


    def calculate_max_latency(self, solution):
        max_latency_per_request = [0] * self.num_requests

        for idx, value in enumerate(solution.variables):
            request_idx = self.mapping[idx]
            dc_idx, _, _ = self.decode(value)
            dc = self.datacenters[dc_idx]
            request_location = self.requests_location[request_idx]
            
            latency = self.latency_lookup[(request_location, dc)]
            if latency > max_latency_per_request[request_idx]:
                max_latency_per_request[request_idx] = latency

        return sum(max_latency_per_request)

    def calculate_qos(self, solution):
        total_interruption = 0
        
        for idx, value in enumerate(solution.variables):
            dc_idx, instance_idx, price_idx = self.decode(value)
            dc = self.datacenters[dc_idx]
            instance = self.instances[instance_idx]
            request_idx = self.mapping[idx]

            if price_idx == 2:  # Spot
                interruption_rate = self.interrupt_rate[(dc, instance)]
                total_interruption += interruption_rate / self.replicas[request_idx]

        qos = total_interruption / self.num_requests
        return qos

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        max_latency = self.calculate_max_latency(solution)
        total_cost, cpu_violations, ram_violations, _ = self.calculate_costs(solution)
        qos = self.calculate_qos(solution)
        
        solution.objectives[0] = max_latency
        solution.objectives[1] = total_cost
        solution.objectives[2] = qos
        
        # Imposta i vincoli come violazioni (più violazioni => peggio è)
        solution.constraints[0] = self.check_number_of_replicas(solution)
        solution.constraints[1] = cpu_violations
        solution.constraints[2] = ram_violations
        
        return solution


    def check_number_of_replicas(self, solution):
        replicas_count = [0] * self.num_requests
        replicas_penalty = [0] * self.num_requests
        for idx, value in enumerate(solution.variables):
            request_idx = self.mapping[idx]
            replicas_count[request_idx] += 1

        for idx_request in range(self.num_requests):
            replicas_penalty[idx_request] = max(0, self.replicas[idx_request] - replicas_count[idx_request])

        return sum(replicas_penalty)

    def number_of_constraints(self):
        return 3

    def name(self):
        return "MultiCluster"
