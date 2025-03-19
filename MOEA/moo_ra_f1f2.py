import numpy as np
import random
import math
import pandas as pd
import os

from jmetal.core.problem import IntegerProblem, FloatProblem
from jmetal.core.solution import IntegerSolution, FloatSolution


class MooRaF1F2(IntegerProblem):
    """Multi-Cluster problem"""

    def __init__(self):
        super(MooRaF1F2, self).__init__()
        self.load_requests_data("./cnsm_data/Services.csv")
        self.load_instances_data(
            "./cnsm_data/pricing.csv", "./cnsm_data/AWS_EC2_Latency.csv"
        )
        print(
            f"Number of datacenters: {self.num_datacenters} and number of requests: {self.num_requests}"
        )

        # Calcola il numero totale di repliche
        self.num_replicas = sum(self.replicas.values())
        self.mapping = self.create_replica_mapping()

        self.num_combinations = self.num_datacenters * 6 * 3  # see moo_ra_3

        print(f"Number of replicas: {self.num_replicas}")
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]

        # we need to map as variables also the starting time of each requests
        # the variables will be a vector in which the first part indicate the timing of each request (length will be num_requests)
        # and the second part indicate the information about replicas

        # Requests can be activated from 0 to 150
        self.lower_bound = [0 for _ in range(self.num_requests)] + [
            0 for _ in range(self.num_replicas)
        ]
        self.upper_bound = [150 for _ in range(self.num_requests)] + [
            self.num_combinations - 1 for _ in range(self.num_replicas)
        ]

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

        self.requests_use_case = {}
        self.requests_cpu = {}
        self.requests_ram = {}
        self.requests_gpu = {}
        self.requests_duration = {}
        self.requests_location = {}
        self.replicas = {}
        self.requests_latency = {}

        for idx, row in self.services.iterrows():
            self.requests_use_case[idx] = row["usecase"]
            self.requests_cpu[idx] = row["cpu"]
            self.requests_ram[idx] = row["ram"]
            self.requests_gpu[idx] = row["gpu"]
            self.requests_duration[idx] = row["duration"]
            self.requests_location[idx] = row["location"]
            self.replicas[idx] = row["replicas"]
            self.requests_latency[idx] = row["latency"]

        self.num_requests = len(self.requests_cpu)

    def load_instances_data(self, pricing_file, latency_file):
        self.pricing = pd.read_csv(pricing_file)
        self.latency = pd.read_csv(latency_file)

        self.datacenters = self.pricing["DataCenter"].unique()
        self.instances = self.pricing["Instance"].unique()

        # Instances cost and capacity data
        self.cost_on_demand = {}
        self.cost_reserved = {}
        self.cost_spot = {}
        self.cpu_capacity = {}
        self.ram_capacity = {}
        self.gpu_capacity = {}
        self.interrupt_rate = {}

        for idx, row in self.pricing.iterrows():
            dc_instance = (row["DataCenter"], row["Instance"])
            self.cost_on_demand[dc_instance] = row["On-Demand"]
            self.cost_reserved[dc_instance] = row["Reserved"]
            self.cost_spot[dc_instance] = row["Spot"]
            self.cpu_capacity[dc_instance] = row["vCPU"]
            self.ram_capacity[dc_instance] = row["RAM"]
            self.gpu_capacity[dc_instance] = row["GPU"]
            self.interrupt_rate[dc_instance] = row["InterruptFrequency"] / 100

        # Latency dictionary
        self.latency_lookup = {}
        for idx, row in self.latency.iterrows():
            from_dc = row["DataCenter"]
            for dc in self.datacenters:
                self.latency_lookup[(from_dc, dc)] = row[dc]

        self.num_datacenters = len(self.datacenters)

    def encode(self, dc_idx, instance_idx, price_idx):
        return dc_idx * 18 + instance_idx * 3 + price_idx

    def decode(self, value):
        dc_idx = value // 18
        instance_idx = (value % 18) // 3
        price_idx = value % 3
        return dc_idx, instance_idx, price_idx

    def calculate_costs(self, solution):
        total_costs = {'On-Demand': 0, 'Reserved': 0, 'Spot': 0}
        instance_usage = {}  # Track instance usage for each bin
        # Initialize violations
        cpu_violations = 0
        ram_violations = 0
        gpu_violations = 0

        # Since here we are mapping starting time as well, what we need to do is to
        #  exclude the first part of the variables containing the starting time
        svariables = solution.variables[self.num_requests:]

        # then to calculate cost we need to use the following code to map replicas to instance
        # Once we know wich replicas are mapped to which instance we can calculate the cost
        # To do so we need to get as timing we need to keep instance open from the request 
        # with the starting time lower than all the others to the request with the starting
        #  time higher than all the others + duration

        latency_violations_before = self.latency_violations(solution)

        # keep track of how time long is the time - horizon of each instance
        min_start_time = min(solution.variables[:self.num_requests])
        max_end_time = 0
        for i in range(self.num_requests):
            et = solution.variables[i] + self.requests_duration[i]
            if et > max_end_time:
                max_end_time = et
        total_time = max_end_time - min_start_time

        check_variables = []
        for idx, value in enumerate(svariables):
            dc_idx, instance_idx, price_idx = self.decode(value)
            dc = self.datacenters[int(dc_idx)]
            instance = self.instances[int(instance_idx)]
            request_idx = self.mapping[idx]

            cpu_required = self.requests_cpu[request_idx]
            ram_required = self.requests_ram[request_idx]
            gpu_required = self.requests_gpu[request_idx]
            latency_required = self.requests_latency[request_idx]
            request_location = self.requests_location[request_idx]
            rd = self.requests_duration[request_idx]
            # we need to find the starting time of the request
            starting_time = solution.variables[request_idx]
            ending_time = starting_time + rd

            latency = self.latency_lookup[(request_location, dc)]
            # the first requirement to check will be latency here
            # so we are going to repair the solution if the latency is not respected
            while latency > latency_required:
                dc_idx = (dc_idx + 1) % self.num_datacenters
                dc = self.datacenters[int(dc_idx)]
                #print(dc_idx, dc, latency, latency_required)
                latency = self.latency_lookup[(request_location, dc)]

            # temporary variables to keep track of the current instance
            dc_instance_key = (dc, instance, price_idx)
            alg_allocation = dc_instance_key
            instance_alg_idx = instance_idx

            cpu_capacity = self.cpu_capacity[(dc, instance)]
            ram_capacity = self.ram_capacity[(dc, instance)]
            gpu_capacity = self.gpu_capacity[(dc, instance)]

            while cpu_capacity < cpu_required or ram_capacity < ram_required or gpu_capacity < gpu_required:
                # try with a different BIN (bigger bins is better)
                instance_alg_idx = (instance_alg_idx + 1) % 6
                instance_alg = self.instances[instance_alg_idx]
                alg_allocation = (dc, instance_alg, price_idx)
                #print(f'instance_alg_idx: {instance_alg_idx}, Instance: {instance_alg}, instance_alg: {alg_allocation}')
                #print(f'Before was {dc_instance_key}')
                cpu_capacity = self.cpu_capacity[(dc, instance_alg)]
                ram_capacity = self.ram_capacity[(dc, instance_alg)]
                gpu_capacity = self.gpu_capacity[(dc, instance_alg)]


            if alg_allocation not in instance_usage:
                # cpu and ram are resetted to 0 each time a new instance/bin is opened
                # here we calculate the duration of the request, for each instance is the time this should be open
                instance_usage[alg_allocation] = [{'cpu': cpu_required, 'ram': ram_required, 
                                                   'gpu': gpu_required, 'active': [starting_time, ending_time]}]
                
            else:
                # Check if there are existing bins for those are already open to contain the request
                allocated = False
                for bin_idx, bin_values in enumerate(instance_usage[alg_allocation]):
                    if bin_values['active'][1] <= starting_time:
                        # Check if the current request can be contained in the bin
                        if bin_values['cpu'] + cpu_required <= cpu_capacity and \
                        bin_values['ram'] + ram_required <= ram_capacity and \
                        bin_values['gpu'] + gpu_required <= gpu_capacity:
                        # Check if the current request can be contained in the bin
                            bin_values['cpu'] += cpu_required
                            bin_values['ram'] += ram_required
                            bin_values['gpu'] += gpu_required
                            bin_values['active'][1] = ending_time
                            allocated = True
                        break
                if not allocated:
                    # Open a new bin for the instance
                    instance_usage[alg_allocation].append({'cpu': cpu_required, 'ram': ram_required,
                                                          'gpu': gpu_required, 'active': [starting_time, ending_time]})

            solution.variables[self.num_requests + idx] = self.encode(dc_idx, instance_alg_idx, price_idx)
            check_variables.append(self.encode(dc_idx, instance_alg_idx, price_idx))
                            
        
        # Calculate total cost based on the costs of each instance considered its active period
        for instance_key, values in instance_usage.items():
            for idx, bin in enumerate(values):
                dc, instance, price_idx = instance_key
                s = bin['active'][0]
                e = bin['active'][1]
                activation_time = e - s
                if price_idx == 0:
                    total_costs['On-Demand'] += self.cost_on_demand[(dc, instance)] * activation_time
                elif price_idx == 1:
                    # Reserved instances are open from time 0 to max duration time
                    total_costs['Reserved'] += self.cost_reserved[(dc, instance)] * total_time
                elif price_idx == 2:
                    total_costs['Spot'] += self.cost_spot[(dc, instance)] * activation_time
                # Check also capacity violations
                if bin['cpu'] > self.cpu_capacity[(dc, instance)]:
                    cpu_violations += bin['cpu'] - self.cpu_capacity[(dc, instance)]
                if bin['ram'] > self.ram_capacity[(dc, instance)]:
                    ram_violations += bin['ram'] - self.ram_capacity[(dc, instance)]
                if bin['gpu'] > self.gpu_capacity[(dc, instance)]:
                    gpu_violations += bin['gpu'] - self.gpu_capacity[(dc, instance)]
        
        total_cost = sum(total_costs.values())
        total_costs['total_cost'] = total_cost

        latency_violations = self.latency_violations(solution)
        if (latency_violations > 0):
            print(f'Latency violations before: {latency_violations_before} after: {latency_violations}, process_id: {os.getpid()}')
            print(f'Variables: {solution.variables[self.num_requests:]}')

        return total_cost, cpu_violations, ram_violations, gpu_violations, instance_usage

    def calculate_max_latency(self, solution):
        max_latency_per_request = [0] * self.num_requests
        # exclude the timing
        svariables = solution.variables[self.num_requests :]
        for idx, value in enumerate(svariables):
            request_idx = self.mapping[idx]
            dc_idx, _, _ = self.decode(value)
            dc = self.datacenters[dc_idx]
            request_location = self.requests_location[request_idx]

            latency = self.latency_lookup[(request_location, dc)]
            if latency > max_latency_per_request[request_idx]:
                max_latency_per_request[request_idx] = latency

        # return the average max_latency per request
        return sum(max_latency_per_request) / self.num_requests

    # This is an additional constraint on latency --
    # each request specified a maximum latency which should not be exceeded
    def latency_violations(self, solution):
        latency_violations = 0
        # exclude the timing
        svariables = solution.variables[self.num_requests :]
        for idx, value in enumerate(svariables):
            request_idx = self.mapping[idx]
            dc_idx, _, _ = self.decode(value)
            dc = self.datacenters[dc_idx]
            request_location = self.requests_location[request_idx]

            latency = self.latency_lookup[(request_location, dc)]
            if latency > self.requests_latency[request_idx]:
                latency_violations += latency - self.requests_latency[request_idx]

        return latency_violations

    def calculate_qos(self, solution):
        total_interruption = 0
        svariables = solution.variables[self.num_requests :]
        for idx, value in enumerate(svariables):
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
        tc, cpu_violations, ram_violations, gpu_violations, _ = self.calculate_costs(
            solution
        )
        # qos = self.calculate_qos(solution)
        latency_violations = self.latency_violations(solution)

        solution.objectives[0] = max_latency
        solution.objectives[1] = tc

        # Imposta i vincoli come violazioni (più violazioni => peggio è)
        solution.constraints[0] = self.check_number_of_replicas(solution)
        solution.constraints[1] = cpu_violations
        solution.constraints[2] = ram_violations
        solution.constraints[3] = gpu_violations
        solution.constraints[4] = latency_violations

        return solution

    def check_number_of_replicas(self, solution):
        replicas_count = [0] * self.num_requests
        replicas_penalty = [0] * self.num_requests
        # exclude the timing
        svariables = solution.variables[self.num_requests :]
        for idx, value in enumerate(svariables):
            request_idx = self.mapping[idx]
            replicas_count[request_idx] += 1

        for idx_request in range(self.num_requests):
            replicas_penalty[idx_request] = max(
                0, self.replicas[idx_request] - replicas_count[idx_request]
            )

        return sum(replicas_penalty)

    def number_of_constraints(self):
        return 5

    def name(self):
        return "Moo_Ra_F1F2"
