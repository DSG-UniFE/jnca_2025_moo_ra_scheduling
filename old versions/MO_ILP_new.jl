using Markdown
using InteractiveUtils

using JuMP, DataFrames, CSV, Gurobi, Plots

import MultiObjectiveAlgorithms as MOA

import MathOptInterface as MOI

function load_data(latency_file::String, instance_file::String, service_file::String)
    latency_df = CSV.read(latency_file, DataFrame)
    instance_df = CSV.read(instance_file, DataFrame)
    service_df = CSV.read(service_file, DataFrame)
    
    DC = latency_df.DataCenter
    I = Set(instance_df.Instance)
    D = 1:nrow(service_df)
    T = 1:100 # Assuming T is 100, adjust as needed

    l = Dict((i, dc) => latency_df[findfirst(==(row.location), latency_df.DataCenter), dc] 
             for (i, row) in enumerate(eachrow(service_df)), dc in DC)
    
    CO = Dict((row.Instance, row.DataCenter) => row."On-Demand" for row in eachrow(instance_df))
    CR = Dict((row.Instance, row.DataCenter) => row.Reserved for row in eachrow(instance_df))
    CS = Dict((row.Instance, row.DataCenter) => row.Spot for row in eachrow(instance_df))
    
    C = Dict((row.Instance, row.DataCenter) => row.vCPU for row in eachrow(instance_df))
    M = Dict((row.Instance, row.DataCenter) => row.RAM for row in eachrow(instance_df))
    
    f = Dict((row.Instance, row.DataCenter) => row.InterruptFrequency / 100 for row in eachrow(instance_df))
    
    c = Dict(i => row.cpu for (i, row) in enumerate(eachrow(service_df)))
    m = Dict(i => row.ram for (i, row) in enumerate(eachrow(service_df)))
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

    return DC, I, D, T, l, CO, CR, CS, C, M, f, c, m, R, Ir, Io, Is, duration, N, Dt
end

function create_model(DC, I, D, T, l, CO, CR, CS, C, M, f, c, m, R, Ir, Io, Is, Dt, N, duration)
	model = Model(Gurobi.Optimizer)
    #model = Model(() -> MOA.Optimizer(Gurobi.Optimizer))
	#set_attribute(model, MOA.Algorithm(), MOA.DominguezRios())
	set_attribute(model, MOI.TimeLimitSec(), 1000)
	set_optimizer_attribute(model, "MIPGap", 0.05)  # 5% gap
	set_optimizer_attribute(model, "SolFiles", "solution_")  # Saves solutions periodically
	set_optimizer_attribute(model, "Threads", 4)
	
    @variable(model, r[j in I, k in DC, n in 1:N], Bin)
    @variable(model, o[t in T, j in I, k in DC, n in 1:N], Bin)
    @variable(model, s[t in T, j in I, k in DC, n in 1:N], Bin)
    @variable(model, a[t in T, i in D], Bin)
    @variable(model, x[t in T, i in D, j in I, k in DC, r in 1:R[i], n in 1:N], Bin)
	@variable(model, y[t in T, i in D, j in I, k in DC, r in 1:R[i], n in 1:N], Bin)
	@variable(model, z[t in T, i in D, j in I, k in DC, r in 1:R[i], n in 1:N], Bin)
    @variable(model, d[i in D, k in DC], Bin)
    @variable(model, 0 <= L[i in D] <= 500)

    @constraint(model, [t in T, j in Ir, k in DC, n in 1:N], 
        sum(x[tp,i,j,k,r,n] * c[i] for (i, tp) in Dt[t], r in 1:R[i]) <= r[j,k,n] * C[j,k])
    @constraint(model, [t in T, j in Io, k in DC, n in 1:N], 
        sum(y[tp,i,j,k,r,n] * c[i] for (i, tp) in Dt[t], r in 1:R[i]) <= o[t,j,k,n] * C[j,k])
    @constraint(model, [t in T, j in Is, k in DC, n in 1:N], 
        sum(z[tp,i,j,k,r,n] * c[i] for (i, tp) in Dt[t], r in 1:R[i]) <= s[t,j,k,n] * C[j,k])

    @constraint(model, [t in T, j in Ir, k in DC, n in 1:N], 
        sum(x[tp,i,j,k,r,n] * m[i] for (i, tp) in Dt[t], r in 1:R[i]) <= r[j,k,n] * M[j,k])
    @constraint(model, [t in T, j in Io, k in DC, n in 1:N], 
        sum(y[tp,i,j,k,r,n] * m[i] for (i, tp) in Dt[t], r in 1:R[i]) <= o[t,j,k,n] * M[j,k])
    @constraint(model, [t in T, j in Is, k in DC, n in 1:N], 
        sum(z[tp,i,j,k,r,n] * m[i] for (i, tp) in Dt[t], r in 1:R[i]) <= s[t,j,k,n] * M[j,k])

    @constraint(model, [i in D, r in 1:R[i]], 
        sum(x[t,i,j,k,r,n] + y[t,i,j,k,r,n] + z[t,i,j,k,r,n] for t in T, j in I, k in DC, n in 1:N) >= 1)

    @constraint(model, [t in T, i in D, j in I, k in DC, r in 1:R[i], n in 1:N], 
        x[t,i,j,k,r,n] + y[t,i,j,k,r,n] + z[t,i,j,k,r,n] <= a[t,i])

    @constraint(model, [i in D], 
        sum(a[t,i] for t in T) == 1)

    @constraint(model, [i in D, k in DC, r in 1:R[i]], 
        d[i,k] >= sum(x[t,i,j,k,r,n] + y[t,i,j,k,r,n] + z[t,i,j,k,r,n] for t in T, j in I, n in 1:N))

    @constraint(model, [i in D, k in DC], 
        L[i] >= d[i,k] * l[i,k])

	@constraint(model, [i in D, j in I, k in DC, r in 1:R[i], n in 1:N, t in length(T)-duration[i]+1:length(T)],
	x[t,i,j,k,r,n] <= 0)

	@constraint(model, [i in D, j in I, k in DC, r in 1:R[i], n in 1:N, t in length(T)-duration[i]+1:length(T)],
	y[t,i,j,k,r,n] <= 0)

	@constraint(model, [i in D, j in I, k in DC, r in 1:R[i], n in 1:N, t in length(T)-duration[i]+1:length(T)],
	z[t,i,j,k,r,n] <= 0)

    @expression(model, f1, 1/length(D) * sum(L[i] for i in D))
    @expression(model, f2, sum(CR[j,k] * r[j,k,n] for j in Ir, k in DC, n in 1:N) + 
                           sum(CO[j,k] * o[t,j,k,n] for t in T, j in Io, k in DC, n in 1:N) + 
                           sum(CS[j,k] * s[t,j,k,n] for t in T, j in Is, k in DC, n in 1:N))
    @expression(model, f3, 1/length(D) * sum(z[t,i,j,k,r,n] * f[j,k] / R[i] 
                                             for i in D, j in Is, k in DC, r in 1:R[i], t in T, n in 1:N))

    #@objective(model, Min, f1) #[f1, f2, f3])

	# solve the model using scalarization ---
	#weights = Dict("f1" => 0.5, "f2" => 0.3, "f3" => 0.2)
	#@objective(model, Min, weights["f1"] * f1 + weights["f2"] * f2 + weights["f3"] * f3)
	# solve all three objectives using DominguezRios
	@objective(model, Min, [f1, f2, f3])

    return model, f1, f2, f3
end

begin
	latency_file = "AWS_EC2_Latency.csv"
	instance_file = "AWS_EC2_Pricing.csv"
	service_file = "Services.csv"
	
	DC, I, D, T, l, CO, CR, CS, C, M, f, c, m, R, Ir, Io, Is, duration, N, Dt = load_data(latency_file, instance_file, service_file)
	
	model, f1, f2, f3 = create_model(DC, I, D, T, l, CO, CR, CS, C, M, f, c, m, R, Ir, Io, Is, Dt, N, duration)
end

optimize!(model)

println(solution_summary(model))

plot = Plots.scatter3d(
	[value(f1; result = i) for i in 1:result_count(model)],
    [value(f2; result = i) for i in 1:result_count(model)],
    [value(f3; result = i) for i in 1:result_count(model)];
    xlabel = "Latency",
    ylabel = "Cost",
	zlabel = "Unavailability"
)

begin
	savefig(plot, "plot.png")
	
	# Save final results
	results_df = DataFrame(
	    Latency = [value(f1; result = i) for i in 1:result_count(model)],
	    Cost = [value(f2; result = i) for i in 1:result_count(model)],
	    Unavailability = [value(f3; result = i) for i in 1:result_count(model)]
	)
	CSV.write("results.csv", results_df)
end
