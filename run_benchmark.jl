using Markdown
using InteractiveUtils

using JuMP, DataFrames, CSV, Gurobi, Plots

import MultiObjectiveAlgorithms as MOA
import MathOptInterface as MOI
using BenchmarkTools

function save_solution_from_summary(model, filename="solution.csv")
    # Get solution summary (this prints details, but doesn't return values)
    println(solution_summary(model))

    # Create DataFrame for variable values
    results = DataFrame(Variable = String[], Value = Float64[])

    # Iterate over all variables in the model
    for v in all_variables(model)
        if !iszero(value(v))
            push!(results, (string(v), value(v)))
        end
    end

    # Save to CSV
    CSV.write(results_path * filename, results)
    println("✅ Solution saved to $filename")
end

function load_data(latency_file::String, instance_file::String, service_file::String)
    latency_df = CSV.read(latency_file, DataFrame)
    instance_df = CSV.read(instance_file, DataFrame)
    service_df = CSV.read(service_file, DataFrame)
    
    DC = latency_df.DataCenter
    I = Set(instance_df.Instance)
    print(I)

    D = 1:nrow(service_df)
    T = 1:100 # Assuming T is 100, adjust as needed

    l = Dict((i, dc) => latency_df[findfirst(==(row.location), latency_df.DataCenter), dc] 
             for (i, row) in enumerate(eachrow(service_df)), dc in DC)
    
    CO = Dict((row.Instance, row.DataCenter) => row."On-Demand" for row in eachrow(instance_df))
    CR = Dict((row.Instance, row.DataCenter) => row.Reserved for row in eachrow(instance_df))
    CS = Dict((row.Instance, row.DataCenter) => row.Spot for row in eachrow(instance_df))
    
    C = Dict((row.Instance, row.DataCenter) => row.vCPU for row in eachrow(instance_df))
    M = Dict((row.Instance, row.DataCenter) => row.RAM for row in eachrow(instance_df))
	G = Dict((row.Instance, row.DataCenter) => row.GPU for row in eachrow(instance_df))
    
    f = Dict((row.Instance, row.DataCenter) => row.InterruptFrequency / 100 for row in eachrow(instance_df))
    
    c = Dict(i => row.cpu for (i, row) in enumerate(eachrow(service_df)))
    m = Dict(i => row.ram for (i, row) in enumerate(eachrow(service_df)))
	g = Dict(i => row.gpu for (i, row) in enumerate(eachrow(service_df)))
	lat_threshold = Dict(i => row.latency for (i, row) in enumerate(eachrow(service_df)))

    R = Dict(i => row.replicas for (i, row) in enumerate(eachrow(service_df)))
    duration = Dict(i => row.duration for (i, row) in enumerate(eachrow(service_df)))

    Ir = I
    Io = I
    Is = I

    N = 10  # Number of instances per instance type

    # Create Dt dictionary
    Dt = Dict{Int, Vector{Tuple{Int, Int}}}()
    for t in T
        Dt[t] = Tuple{Int, Int}[]
        for (i, d) in duration
            for start_t in max(1, t - d + 1):min(t, length(T) - d + 1)
                if start_t + d - 1 >= t
                    push!(Dt[t], (i, start_t))
                end
            end
        end
    end

    return DC, I, D, T, l, CO, CR, CS, C, M, G, f, c, m, g, lat_threshold, R, Ir, Io, Is, duration, N, Dt
end

function create_model(DC, I, D, T, l, CO, CR, CS, C, M, G, f, c, m, g, lat_threshold, R, Ir, Io, Is, Dt, N, duration, gap)
    model = Model(Gurobi.Optimizer)
    # model = Model(() -> MOA.Optimizer(Gurobi.Optimizer))
	# set_attribute(model, MOA.Algorithm(), MOA.DominguezRios())
	set_attribute(model, MOI.TimeLimitSec(), 1000)
	set_optimizer_attribute(model, "MIPGap", gap)
	set_optimizer_attribute(model, "SolFiles", "solution_")  # Saves solutions periodically
	#set_optimizer_attribute(model, "Threads", 4)

    @variable(model, r[j in I, k in DC], Int)
    @variable(model, o[t in T, j in I, k in DC], Int)
    @variable(model, s[t in T, j in I, k in DC], Int)
    @variable(model, a[t in T, i in D], Bin)
    @variable(model, x[t in T, i in D, j in I, k in DC, r in 1:R[i]], Bin)
	@variable(model, y[t in T, i in D, j in I, k in DC, r in 1:R[i]], Bin)
	@variable(model, z[t in T, i in D, j in I, k in DC, r in 1:R[i]], Bin)
    @variable(model, d[i in D, k in DC], Int)
    #@variable(model, 0 <= L[i in D] <= 500)

	# Upper bound constraint for latency threshold
	@variable(model, 0 <= L[i in D] <= lat_threshold[i])

	# CPU Constraints
    @constraint(model, [t in T, j in Ir, k in DC],
        sum(x[tp,i,j,k,r] * c[i] for (i, tp) in Dt[t], r in 1:R[i]) <= r[j,k] * C[j,k])
    @constraint(model, [t in T, j in Io, k in DC],
        sum(y[tp,i,j,k,r] * c[i] for (i, tp) in Dt[t], r in 1:R[i]) <= o[t,j,k] * C[j,k])
    @constraint(model, [t in T, j in Is, k in DC],
        sum(z[tp,i,j,k,r] * c[i] for (i, tp) in Dt[t], r in 1:R[i]) <= s[t,j,k] * C[j,k])

	# Memory Constraints
    @constraint(model, [t in T, j in Ir, k in DC],
        sum(x[tp,i,j,k,r] * m[i] for (i, tp) in Dt[t], r in 1:R[i]) <= r[j,k] * M[j,k])
    @constraint(model, [t in T, j in Io, k in DC],
        sum(y[tp,i,j,k,r] * m[i] for (i, tp) in Dt[t], r in 1:R[i]) <= o[t,j,k] * M[j,k])
    @constraint(model, [t in T, j in Is, k in DC],
        sum(z[tp,i,j,k,r] * m[i] for (i, tp) in Dt[t], r in 1:R[i]) <= s[t,j,k] * M[j,k])

	# GPU Constraints
    @constraint(model, [t in T, j in Ir, k in DC],
        sum(x[tp,i,j,k,r] * g[i] for (i, tp) in Dt[t], r in 1:R[i]) <= r[j,k] * G[j,k])
    @constraint(model, [t in T, j in Io, k in DC],
        sum(y[tp,i,j,k,r] * g[i] for (i, tp) in Dt[t], r in 1:R[i]) <= o[t,j,k] * G[j,k])
    @constraint(model, [t in T, j in Is, k in DC],
        sum(z[tp,i,j,k,r] * g[i] for (i, tp) in Dt[t], r in 1:R[i]) <= s[t,j,k] * G[j,k])

    # one replica should be associated with only one instance
    @constraint(model, [i in D, r in 1:R[i]],
        sum(x[t,i,j,k,r] + y[t,i,j,k,r] + z[t,i,j,k,r] for t in T, j in I, k in DC) >= 1)

    @constraint(model, [t in T, i in D, j in I, k in DC, r in 1:R[i]],
        x[t,i,j,k,r] + y[t,i,j,k,r] + z[t,i,j,k,r] <= a[t,i])

    @constraint(model, [i in D],
        sum(a[t,i] for t in T) == 1)

    @constraint(model, [i in D, k in DC, r in 1:R[i]],
        d[i,k] >= sum(x[t,i,j,k,r] + y[t,i,j,k,r] + z[t,i,j,k,r] for t in T, j in I))

    @constraint(model, [i in D, k in DC],
        L[i] >= d[i,k] * l[i,k])

	@constraint(model, [i in D, j in I, k in DC, r in 1:R[i], t in length(T)-duration[i]+1:length(T)],
	x[t,i,j,k,r] <= 0)

	@constraint(model, [i in D, j in I, k in DC, r in 1:R[i], t in length(T)-duration[i]+1:length(T)],
	y[t,i,j,k,r] <= 0)

	@constraint(model, [i in D, j in I, k in DC, r in 1:R[i], t in length(T)-duration[i]+1:length(T)],
	z[t,i,j,k,r] <= 0)

    @expression(model, f1, 1/length(D) * sum(L[i] for i in D))

    # Reserved instances be used for the entire duration T 
    @expression(model, f2, sum(CR[j,k] * r[j,k] for j in Ir, k in DC) * last(T) +
                           sum(CO[j,k] * o[t,j,k] for t in T, j in Io, k in DC) + 
                           sum(CS[j,k] * s[t,j,k] for t in T, j in Is, k in DC))

    @expression(model, f3, 1/length(D) * sum(z[t,i,j,k,r] * f[j,k] / R[i]
                                             for i in D, j in Is, k in DC, r in 1:R[i], t in T))

    #@objective(model, Min, f1) #[f1, f2, f3])

	# solve all three objectives using DominguezRios
    # Currently variables r, o, s are not bounded, thus DominguezRios will not work
    # Gurobi alone works well with this model (even with larger instances)
    # However, it solves the problem with a linearization of the single objectives

	@objective(model, Min, [f1, f2, f3])

    return model, f1, f2, f3
end

# Define the list of MIP gaps to test
mip_gaps = [0.0] # 0%, 5%, 10%, 25%

# Initialize an empty DataFrame to store results
results_df = DataFrame(
    MIP_Gap = Float64[],
    Load_Time = Float64[],
    Creation_Time = Float64[],
    Execution_Time = Float64[],
    Latency = Float64[],
    Cost = Float64[],
    Unavailability = Float64[]
)

latency_file = "AWS/latency.csv"
instance_file = "AWS/pricing.csv"

# Change here according to the usecase you want to test
usecase = "smartcity50" # "smartcity", "iiot", "ai", "vr"

# Paths
service_file = "services/types/" * usecase * ".csv"
results_path = "results/usecase/" * usecase * "/f1_f2_f3/"

# Store all results in a DataFrame
benchmark_results = DataFrame()

begin
	DC, I, D, T, l, CO, CR, CS, C, M, G, f, c, m, g, lat_threshold, R, Ir, Io, Is, duration, N, Dt = load_data(latency_file, instance_file, service_file)

    model, f1, f2, f3 = create_model(DC, I, D, T, l, CO, CR, CS, C, M, G, f, c, m, g, lat_threshold, R, Ir, Io, Is, Dt, N, duration, 0.0)
end

for gap in mip_gaps
    println("\n🔍 Running benchmark with MIP Gap: ", gap)

	# Change MIP gap
	set_optimizer_attribute(model, "MIPGap", gap)

    # Benchmark optimization
    benchmark_result = @benchmark optimize!(model)

    # Compute additional metrics
    gc_time = mean(benchmark_result).gctime / 1e6  # GC time in ms
    gc_percentage = (gc_time / (mean(benchmark_result).time / 1e6)) * 100  # GC % of total time
    throughput = 1 / (mean(benchmark_result).time / 1e9)  # Ops per second

    # Store benchmark results
    push!(benchmark_results, (MIP_Gap = gap,
                               Min_Time_ms = minimum(benchmark_result).time / 1e6,
                               Mean_Time_ms = mean(benchmark_result).time / 1e6,
                               Max_Time_ms = maximum(benchmark_result).time / 1e6,
                               Median_Time_ms = median(benchmark_result).time / 1e6,
                               Memory_Allocations = benchmark_result.memory,
                               Memory_Usage_bytes = benchmark_result.allocs,
                               GC_Time_ms = gc_time,
                               GC_Percentage = gc_percentage,
                               Throughput_Ops_sec = throughput))
end

# Save all results to CSV
CSV.write(results_path * "benchmark_results_usecase_$(usecase).csv", benchmark_results)
println("✅ Benchmark results saved to benchmark_results_usecase_$(usecase).csv")
