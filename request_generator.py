import numpy as np
import pandas as pd
import random
from scipy.stats import pareto
import os

MAX_REPLICAS = 6


# Set the parameters for the Pareto distribution
shape = 2.62  # shape parameter (alpha)
scale = (shape - 1) / (100 * shape)  # calculated scale parameter for 100 requests per unit time
scale2 = (shape - 1) / (200 * shape)  # calculated scale for spike in request arrival rate
# Number of requests to generate


# Model also some spikes in the request arrival rate

# Set the vCPU and RAM ranges for the pods for each usecase
vCPU_ranges = [(1, 6), (1, 4), (1, 3), (2, 4)]  # in vCPUs usecases ai, iiot, smartcity, vr  
vRAM_ranges = [(2, 8), (2, 6), (1, 3), (2, 4)]  # in MB usecases ai, iiot, smartcity, vr
gpu_ranges = [(0, 16), (0, 8), (0, 4), (0, 12)] # in GB usecases ai, iiot, smartcity, vr
duration_range = (10, 50) # hours
latency_ranges = [(20, 40), (5, 15), (200, 500), (10, 120)] # ms usecases ai, iiot, smartcity, vr

dcs = ["eu-south-1", "eu-west-2", "us-east-1", "us-west-1", "ap-northeast-3", "ap-south-1"]
usecases = ["ai", "iiot", "smartcity", "vr"]
requests_names = [("preprocessing", "inference", "postprocessing"), ("datacollection", "preprocessing", "anomaly"), ("aggregation", "analysis"), ("capturing", "preprocessing", "decoding", "rendering")]
# Generate interarrival times using Pareto distribution



output_path = "services/types"
os.makedirs(output_path, exist_ok=True)

# For each use case, generates 50 and 100 requests csv file
for idx, usecase in enumerate(usecases):
    for num_requests in [50, 100]:
        interarrival_times = pareto.rvs(shape, scale=scale, size=num_requests)
        requests = []
        for i in range(num_requests):
            request_time = np.sum(interarrival_times[:i+1])
            num_replicas = random.randint(1, MAX_REPLICAS)  # Number of pods
            usecase = usecases[idx]
            name = random.choice(requests_names[idx])
            vCPU = random.randint(vCPU_ranges[idx][0], vCPU_ranges[idx][1])
            vRAM = random.randint(vRAM_ranges[idx][0], vRAM_ranges[idx][1])
            gpu = random.randint(gpu_ranges[idx][0], gpu_ranges[idx][1])
            duration = random.randint(duration_range[0], duration_range[1])
            location = random.choice(dcs)
            latency = random.randint(latency_ranges[idx][0], latency_ranges[idx][1])
            request = {
                "usecase": usecase,
                "name": name,
                "cpu": vCPU,
                "gpu": gpu,
                "ram": vRAM,
                "duration": duration,
                "location": location,
                "replicas": num_replicas,
                "latency": latency
            }
            requests.append(request)
        
        df_requests = pd.DataFrame(requests)
        filename = os.path.join(output_path, f"{usecase}{num_requests}.csv")
        df_requests.to_csv(filename, index=False)

'''
# Generate requests
for i in range(num_requests):
    
    vCPU = random.randint(vCPU_ranges[0], vCPU_ranges[1])
    vRAM = random.randint(vRAM_ranges[0], vRAM_ranges[1])
    gpu = random.randint(gpu_ranges[0], gpu_ranges[1])
    dc = random.choice(dcs)
    latency = random.randint(latency_ranges[0], latency_ranges[1])
    name = random.choice(requests_names)
    usecase = random.choice(usecases)
    
    request_ai = {
        # integer value of the request time so we can use 
        # the index as time-slot index
        #"request_time": int(request_time),
        "usecase": usecase,
        "name": name,
        "cpu": vCPU,
        "gpu": gpu,
        "ram":vRAM,
        "duration": random.randint(duration_range[0], duration_range[1]),
        "location": dc,
        "replicas": num_replicas,
        "latency": latency
    }
    
    requests.append(request_ai)


# add spikes in the request arrival rate
spike_requests = 100 
interarrival_times = pareto.rvs(shape, scale=scale2, size=spike_requests)

lrt = requests[-1]["request_time"]

for i in range(spike_requests):
    request_time = np.sum(interarrival_times[:i+1])
    num_pods = random.randint(1, MAX_PODS)  # Number of pods
    vCPU = random.randint(vCPU_range[0], vCPU_range[1])
    RAM = random.choice(RAM_options)
    
    request = {
        # integer value of the request time so we can use 
        # the index as time-slot index
        "request_time": int(lrt + request_time),
        "num_pods": num_pods,
        "vCPU": vCPU,
        "RAM": RAM
    }
    
    requests.append(request)

# additional requests with normal scale
num_requests = 300 
interarrival_times = pareto.rvs(shape, scale=scale, size=num_requests)
lrt = requests[-1]["request_time"]
for i in range(num_requests):
    request_time = np.sum(interarrival_times[:i+1])
    num_pods = random.randint(1, MAX_PODS)  # Number of pods
    vCPU = random.randint(vCPU_range[0], vCPU_range[1])
    RAM = random.choice(RAM_options)
    
    request = {
        # integer value of the request time so we can use 
        # the index as time-slot index
        "request_time": int(lrt + request_time),
        "num_pods": num_pods,
        "vCPU": vCPU,
        "RAM": RAM
    }
    
    requests.append(request)


# Convert the requests to a DataFrame for easy manipulation and visualization
df_requests = pd.DataFrame(requests)

# Display the generated requests
#print(df_requests)

# Optionally save the requests to a CSV file
df_requests.to_csv('services_ai.csv', index=False)
'''
