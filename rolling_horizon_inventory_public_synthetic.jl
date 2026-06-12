using Random
using JuMP
using HiGHS
using Statistics

# ==============================================================================
# Rolling-horizon inventory optimization under demand and lead-time uncertainty
# ==============================================================================
# Thesis application: one SKU, one regional warehouse, monthly planning.
#
# The framework combines:
#   1. a deterministic LP solved at each rolling decision month,
#   2. stochastic simulation of demand and shipment arrivals,
#   3. Monte Carlo evaluation of alternative inventory policies.
# ==============================================================================

# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------

"""
Shipment lot currently in transit.

A shipment is treated as indivisible: it either fully arrives in a period or remains
fully in transit.
"""
struct Shipment
    region::Int
    ship_month::Int
    qty::Float64
end

"""Policy parameters evaluated in the simulation experiments."""
struct PolicyConfig
    name::String
    safety_stock::Float64
    demand_uplift::Float64
    lead_time_policy::Symbol   # :expected or :worst_case
end

"""Cost and simulation parameters shared across experiments."""
struct SimulationParams
    cogs::Float64
    holding_percent::Float64
    shortage_cost::Float64
    shipment_cost_per_unit::Float64
    demand_bias_min::Float64
    demand_bias_max::Float64
    n_regions::Int
end

const DEFAULT_PARAMS = SimulationParams(
    30.0,    # cogs
    0.025,   # monthly holding cost as % of COGS
    3.0,     # shortage penalty per unit
    1.5,     # shipment cost per unit
    -0.30,   # minimum demand forecast error
    0.30,    # maximum demand forecast error
    1,       # number of regions
)

# ------------------------------------------------------------------------------
# Lead-time probabilities and stochastic arrivals
# ------------------------------------------------------------------------------

"""
    hazard_to_pmf(hazard)

Convert conditional arrival hazards into an unconditional lead-time PMF.

`hazard[a] = P(L = a | L >= a)`, where `L` is the lead time in months.
"""
function hazard_to_pmf(hazard::Dict{Int,Float64})::Dict{Int,Float64}
    ages = sort(collect(keys(hazard)))
    isempty(ages) && error("Lead-time hazard dictionary cannot be empty.")

    pmf = Dict{Int,Float64}()
    survival_probability = 1.0

    for age in ages
        h = hazard[age]
        0.0 <= h <= 1.0 || error("Hazard probability at age $age must be in [0, 1].")

        pmf[age] = survival_probability * h
        survival_probability *= 1.0 - h
    end

    return pmf
end

"""
    build_lead_time_pmf(hazard, policy)

Return the lead-time PMF used inside the optimization model.

- `:expected`: use the empirical PMF implied by the hazard rates.
- `:worst_case`: place all probability mass on the longest lead time.
"""
function build_lead_time_pmf(
    hazard::Dict{Int,Float64},
    policy::Symbol,
)::Dict{Int,Float64}
    if policy == :expected
        return hazard_to_pmf(hazard)
    elseif policy == :worst_case
        return Dict(maximum(keys(hazard)) => 1.0)
    else
        error("Unknown lead_time_policy=$policy. Use :expected or :worst_case.")
    end
end

"""
    realize_arrivals!(pipeline, current_month, n_regions, hazard; rng)

Simulate actual arrivals for all in-transit shipments in the current month.
"""
function realize_arrivals!(
    pipeline::Vector{Shipment},
    current_month::Int,
    n_regions::Int,
    hazard::Dict{Int,Float64};
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::Vector{Float64}
    arrivals = zeros(n_regions)
    remaining = Shipment[]

    for shipment in pipeline
        shipment_age = current_month - shipment.ship_month
        arrival_probability = get(hazard, shipment_age, 0.0)

        if rand(rng) < arrival_probability
            arrivals[shipment.region] += shipment.qty
        else
            push!(remaining, shipment)
        end
    end

    empty!(pipeline)
    append!(pipeline, remaining)
    return arrivals
end

"""
    add_expected_arrivals_from_existing_shipments!(Aexp, pipeline, horizon, start_t, pmf)

Add conditional expected arrivals from shipments already in transit at the beginning
of the rolling-horizon optimization step.
"""
function add_expected_arrivals_from_existing_shipments!(
    Aexp::Matrix{Float64},
    pipeline::Vector{Shipment},
    horizon::UnitRange{Int},
    start_t::Int,
    pmf::Dict{Int,Float64},
)::Nothing
    for shipment in pipeline
        current_age = start_t - shipment.ship_month
        remaining_mass = sum(prob for (age, prob) in pmf if age >= current_age)

        remaining_mass <= 1e-12 && continue

        for (local_time, calendar_month) in enumerate(horizon)
            target_age = calendar_month - shipment.ship_month

            if target_age >= current_age && haskey(pmf, target_age)
                conditional_probability = pmf[target_age] / remaining_mass
                Aexp[shipment.region, local_time] += shipment.qty * conditional_probability
            end
        end
    end

    return nothing
end

# ------------------------------------------------------------------------------
# Forecast and demand functions
# ------------------------------------------------------------------------------

"""
    get_forecast_window(forecast_versions, start_t, horizon)

Return the forecast vector available at rolling decision month `start_t`.

Column convention:
- column 12 = forecast made 1 month before target month,
- column 11 = forecast made 2 months before target month,
- ...
- column 1  = forecast made 12 months before target month.
"""
function get_forecast_window(
    forecast_versions::Matrix{Float64},
    start_t::Int,
    horizon::UnitRange{Int},
)::Vector{Float64}
    max_horizon = size(forecast_versions, 2)
    forecast = zeros(length(horizon))

    for (local_index, target_month) in enumerate(horizon)
        months_ahead = target_month - start_t + 1
        forecast_column = max_horizon - months_ahead + 1

        1 <= forecast_column <= max_horizon || error(
            "Forecast column out of range for start_t=$start_t, target_month=$target_month.",
        )

        forecast[local_index] = forecast_versions[target_month, forecast_column]
    end

    return forecast
end

"""
    realize_demand(forecast_month, n_regions; bias_min, bias_max, rng)

Simulate realized demand as `forecast * (1 + bias)`, where the bias is uniformly
drawn from `[bias_min, bias_max]`.
"""
function realize_demand(
    forecast_month::Float64,
    n_regions::Int;
    bias_min::Float64,
    bias_max::Float64,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::Vector{Float64}
    demand = zeros(n_regions)

    for region in 1:n_regions
        bias = rand(rng) * (bias_max - bias_min) + bias_min
        demand[region] = forecast_month * (1.0 + bias)
    end

    return demand
end

# ------------------------------------------------------------------------------
# Rolling-horizon optimization and simulation
# ------------------------------------------------------------------------------

"""
    solve_rolling_step(...)

Solve the LP at one rolling decision month and return the planned shipment matrix.
Only the first-column decision is implemented by the simulation loop.
"""
function solve_rolling_step(
    inventory_state::Vector{Float64},
    backorder_state::Vector{Float64},
    demand_forecast::Vector{Float64},
    arrivals_existing::Matrix{Float64},
    horizon::UnitRange{Int},
    pmf::Dict{Int,Float64},
    policy::PolicyConfig,
    params::SimulationParams,
)::Matrix{Float64}
    n_regions = params.n_regions
    n_periods = length(horizon)
    regions = 1:n_regions
    periods = 1:n_periods

    optimization_demand = [
        demand_forecast[t] * (1.0 + policy.demand_uplift)
        for _ in regions, t in periods
    ]

    holding_cost_per_unit = params.cogs * params.holding_percent

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[regions, periods] >= 0)
    @variable(model, I[regions, periods] >= 0)
    @variable(model, B[regions, periods] >= 0)

    for r in regions, t in periods
        previous_inventory = t == 1 ? inventory_state[r] : I[r, t - 1]
        previous_backorders = t == 1 ? backorder_state[r] : B[r, t - 1]

        expected_arrivals_new = sum(
            get(pmf, horizon[t] - horizon[j], 0.0) * x[r, j]
            for j in periods if horizon[j] < horizon[t];
            init = 0.0,
        )

        @constraint(model,
            I[r, t] == previous_inventory
                       - previous_backorders
                       + arrivals_existing[r, t]
                       + expected_arrivals_new
                       - optimization_demand[r, t]
                       + B[r, t]
        )

        @constraint(model, I[r, t] >= policy.safety_stock)
    end

    @objective(model, Min,
        params.shipment_cost_per_unit * sum(x[r, t] for r in regions, t in periods) +
        sum(
            holding_cost_per_unit * I[r, t] + params.shortage_cost * B[r, t]
            for r in regions, t in periods
        )
    )

    optimize!(model)

    status = termination_status(model)
    string(status) == "OPTIMAL" || error("Optimization failed with status: $status")

    return value.(x)
end

"""
    run_rolling_horizon(...)

Run one stochastic trajectory of the rolling-horizon simulation.
"""
function run_rolling_horizon(;
    forecast_versions::Matrix{Float64},
    hazard::Dict{Int,Float64},
    Ir_init::Vector{Float64},
    Br_init::Vector{Float64},
    shipments_in_transit_init::Vector{Shipment},
    policy::PolicyConfig,
    params::SimulationParams = DEFAULT_PARAMS,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    full_horizon = 1:size(forecast_versions, 1)
    max_forecast_horizon = size(forecast_versions, 2)
    pmf = build_lead_time_pmf(hazard, policy.lead_time_policy)

    inventory_state = copy(Ir_init)
    backorder_state = copy(Br_init)
    pipeline = copy(shipments_in_transit_init)

    n_regions = params.n_regions
    n_months = length(full_horizon)

    x_impl = zeros(n_regions, n_months)
    inventory_impl = zeros(n_regions, n_months)
    backorder_impl = zeros(n_regions, n_months)
    arrivals_realized = zeros(n_regions, n_months)
    demand_realized = zeros(n_regions, n_months)
    forecast_used = fill(NaN, n_months, n_months)

    for start_t in full_horizon
        arrivals_now = realize_arrivals!(
            pipeline,
            start_t,
            n_regions,
            hazard;
            rng = rng,
        )
        arrivals_realized[:, start_t] .= arrivals_now

        horizon_end = min(last(full_horizon), start_t + max_forecast_horizon - 1)
        horizon = start_t:horizon_end

        demand_forecast = get_forecast_window(forecast_versions, start_t, horizon)
        for (local_index, target_month) in enumerate(horizon)
            forecast_used[start_t, target_month] = demand_forecast[local_index]
        end

        arrivals_existing = zeros(n_regions, length(horizon))
        add_expected_arrivals_from_existing_shipments!(
            arrivals_existing,
            pipeline,
            horizon,
            start_t,
            pmf,
        )
        arrivals_existing[:, 1] .+= arrivals_now

        planned_shipments = solve_rolling_step(
            inventory_state,
            backorder_state,
            demand_forecast,
            arrivals_existing,
            horizon,
            pmf,
            policy,
            params,
        )

        demand_now = realize_demand(
            demand_forecast[1],
            n_regions;
            bias_min = params.demand_bias_min,
            bias_max = params.demand_bias_max,
            rng = rng,
        )
        demand_realized[:, start_t] .= demand_now

        for r in 1:n_regions
            implemented_shipment = planned_shipments[r, 1]
            x_impl[r, start_t] = implemented_shipment

            net_inventory = inventory_state[r] - backorder_state[r] + arrivals_now[r] - demand_now[r]

            if net_inventory >= 0.0
                inventory_state[r] = net_inventory
                backorder_state[r] = 0.0
            else
                inventory_state[r] = 0.0
                backorder_state[r] = -net_inventory
            end

            inventory_impl[r, start_t] = inventory_state[r]
            backorder_impl[r, start_t] = backorder_state[r]

            implemented_shipment > 1e-9 && push!(pipeline, Shipment(r, start_t, implemented_shipment))
        end
    end

    return (
        x_impl = x_impl,
        Ir_impl = inventory_impl,
        B_impl = backorder_impl,
        arrivals_realized = arrivals_realized,
        demand_realized = demand_realized,
        forecast_used = forecast_used,
        shipments_in_transit_final = pipeline,
    )
end

# ------------------------------------------------------------------------------
# KPI computation and Monte Carlo evaluation
# ------------------------------------------------------------------------------

"""Compute performance indicators on a selected evaluation window."""
function compute_kpis(
    result;
    eval_start::Int = 6,
    eval_end::Int = 18,
    params::SimulationParams = DEFAULT_PARAMS,
)
    eval_months = eval_start:eval_end
    holding_cost_per_unit = params.cogs * params.holding_percent

    demand = result.demand_realized[1, eval_months]
    backorders = result.B_impl[1, eval_months]
    inventory = result.Ir_impl[1, eval_months]
    shipments = result.x_impl[1, eval_months]

    total_demand = sum(demand)
    total_backorders = sum(backorders)

    cfr = total_demand > 1e-9 ? 1.0 - total_backorders / total_demand : 1.0
    cfr = clamp(cfr, 0.0, 1.0)

    shipping_cost = params.shipment_cost_per_unit * sum(shipments)
    holding_cost = holding_cost_per_unit * sum(inventory)
    shortage_cost = params.shortage_cost * total_backorders
    total_cost = shipping_cost + holding_cost + shortage_cost

    return (
        cfr = cfr,
        avg_inventory = mean(inventory),
        max_inventory = maximum(inventory),
        avg_backorder = mean(backorders),
        total_backorders = total_backorders,
        stockout_indicator = any(backorders .> 1e-6) ? 1.0 : 0.0,
        stockout_months = count(backorders .> 1e-6),
        shipping_cost = shipping_cost,
        holding_cost = holding_cost,
        shortage_cost = shortage_cost,
        total_cost = total_cost,
    )
end

"""Run Monte Carlo simulations for a set of policies."""
function evaluate_policies(
    policies::Vector{PolicyConfig};
    forecast_versions::Matrix{Float64},
    hazard::Dict{Int,Float64},
    Ir_init::Vector{Float64},
    Br_init::Vector{Float64},
    shipments_in_transit_init::Vector{Shipment},
    params::SimulationParams = DEFAULT_PARAMS,
    n_sims::Int = 400,
    eval_start::Int = 6,
    eval_end::Int = 18,
)
    summary = Dict{String,Any}()

    for policy in policies
        println("\nRunning Monte Carlo for: $(policy.name)")

        cfr_samples = Float64[]
        avg_inventory_samples = Float64[]
        max_inventory_samples = Float64[]
        avg_backorder_samples = Float64[]
        total_backorder_samples = Float64[]
        stockout_indicator_samples = Float64[]
        stockout_month_samples = Float64[]
        shipping_cost_samples = Float64[]
        holding_cost_samples = Float64[]
        shortage_cost_samples = Float64[]
        total_cost_samples = Float64[]

        for sim in 1:n_sims
            rng = MersenneTwister(1000 + sim)  # common random numbers across policies

            result = run_rolling_horizon(
                forecast_versions = forecast_versions,
                hazard = hazard,
                Ir_init = Ir_init,
                Br_init = Br_init,
                shipments_in_transit_init = shipments_in_transit_init,
                policy = policy,
                params = params,
                rng = rng,
            )

            kpi = compute_kpis(result; eval_start = eval_start, eval_end = eval_end, params = params)

            push!(cfr_samples, kpi.cfr)
            push!(avg_inventory_samples, kpi.avg_inventory)
            push!(max_inventory_samples, kpi.max_inventory)
            push!(avg_backorder_samples, kpi.avg_backorder)
            push!(total_backorder_samples, kpi.total_backorders)
            push!(stockout_indicator_samples, kpi.stockout_indicator)
            push!(stockout_month_samples, kpi.stockout_months)
            push!(shipping_cost_samples, kpi.shipping_cost)
            push!(holding_cost_samples, kpi.holding_cost)
            push!(shortage_cost_samples, kpi.shortage_cost)
            push!(total_cost_samples, kpi.total_cost)
        end

        summary[policy.name] = (
            policy = policy,
            cfr_samples = cfr_samples,
            avg_inventory_samples = avg_inventory_samples,
            avg_backorder_samples = avg_backorder_samples,
            total_cost_samples = total_cost_samples,
            mean_cfr = mean(cfr_samples),
            std_cfr = std(cfr_samples),
            p05_cfr = quantile(cfr_samples, 0.05),
            p95_cfr = quantile(cfr_samples, 0.95),
            mean_inventory = mean(avg_inventory_samples),
            std_inventory = std(avg_inventory_samples),
            mean_max_inventory = mean(max_inventory_samples),
            mean_backorder = mean(avg_backorder_samples),
            mean_total_backorders = mean(total_backorder_samples),
            prob_stockout = mean(stockout_indicator_samples),
            mean_stockout_months = mean(stockout_month_samples),
            mean_shipping_cost = mean(shipping_cost_samples),
            mean_holding_cost = mean(holding_cost_samples),
            mean_shortage_cost = mean(shortage_cost_samples),
            mean_total_cost = mean(total_cost_samples),
            std_total_cost = std(total_cost_samples),
            p05_total_cost = quantile(total_cost_samples, 0.05),
            p95_total_cost = quantile(total_cost_samples, 0.95),
        )
    end

    return summary
end

"""Print a compact Monte Carlo summary table."""
function print_monte_carlo_summary(summary::Dict{String,Any})::Nothing
    println("\n" * "="^160)
    println("Policy | Mean CFR | Std CFR | P(stockout) | Avg Inventory | Avg Backorder | Total Backorders | Mean Cost | Std Cost")
    println("="^160)

    for policy_name in sort(collect(keys(summary)))
        result = summary[policy_name]
        println(
            rpad(policy_name, 28), " | ",
            rpad(string(round(result.mean_cfr, digits = 4)), 8), " | ",
            rpad(string(round(result.std_cfr, digits = 4)), 7), " | ",
            rpad(string(round(result.prob_stockout, digits = 4)), 11), " | ",
            rpad(string(round(result.mean_inventory, digits = 2)), 13), " | ",
            rpad(string(round(result.mean_backorder, digits = 2)), 13), " | ",
            rpad(string(round(result.mean_total_backorders, digits = 2)), 16), " | ",
            rpad(string(round(result.mean_total_cost, digits = 2)), 9), " | ",
            round(result.std_total_cost, digits = 2),
        )
    end

    return nothing
end

"""Run and print one diagnostic trajectory for one policy."""
function run_diagnostic(
    policy::PolicyConfig;
    forecast_versions::Matrix{Float64},
    hazard::Dict{Int,Float64},
    Ir_init::Vector{Float64},
    Br_init::Vector{Float64},
    shipments_in_transit_init::Vector{Shipment},
    params::SimulationParams = DEFAULT_PARAMS,
    seed::Int = 1234,
    eval_start::Int = 6,
    eval_end::Int = 18,
)
    println("\nDiagnostic run for: $(policy.name)")

    result = run_rolling_horizon(
        forecast_versions = forecast_versions,
        hazard = hazard,
        Ir_init = Ir_init,
        Br_init = Br_init,
        shipments_in_transit_init = shipments_in_transit_init,
        policy = policy,
        params = params,
        rng = MersenneTwister(seed),
    )

    println("\nMonth | Forecast | Demand | Shipment | Arrivals | Inventory | Backorders")
    println("-"^88)

    for t in axes(result.Ir_impl, 2)
        println(
            lpad(t, 5), " | ",
            lpad(round(Int, result.forecast_used[t, t]), 8), " | ",
            lpad(round(Int, result.demand_realized[1, t]), 6), " | ",
            lpad(round(Int, result.x_impl[1, t]), 8), " | ",
            lpad(round(Int, result.arrivals_realized[1, t]), 8), " | ",
            lpad(round(Int, result.Ir_impl[1, t]), 9), " | ",
            lpad(round(Int, result.B_impl[1, t]), 10),
        )
    end

    kpi = compute_kpis(result; eval_start = eval_start, eval_end = eval_end, params = params)

    println("\nDiagnostic summary, months $eval_start--$eval_end")
    println("Customer fill rate:     ", round(kpi.cfr, digits = 4))
    println("Average inventory:      ", round(kpi.avg_inventory, digits = 2))
    println("Maximum inventory:      ", round(kpi.max_inventory, digits = 2))
    println("Total backorders:       ", round(kpi.total_backorders, digits = 2))
    println("Stockout months:        ", kpi.stockout_months)
    println("Total cost:             ", round(kpi.total_cost, digits = 2))
    println("Months with shipments:  ", findall(result.x_impl[1, :] .> 1e-6))
    println("Months with arrivals:   ", findall(result.arrivals_realized[1, :] .> 1e-6))
    println("Months with backorders: ", findall(result.B_impl[1, :] .> 1e-6))

    return result, kpi
end

# ==============================================================================
# 6. Input data
# ==============================================================================

forecast_versions_2024_2025 = [
    15400.00 14750.00 14400.00 14200.00 14550.00 14200.00 14150.00 13900.00 15100.00 13400.00 13750.00 13350.00;
    20650.00 19750.00 19650.00 19450.00 19000.00 18850.00 18500.00 18400.00 17900.00 16700.00 19450.00 18000.00;
    38650.00 40700.00 40500.00 41850.00 44000.00 44200.00 44950.00 46150.00 50550.00 41800.00 45500.00 52350.00;
    18100.00 17350.00 17450.00 17000.00 17200.00 16600.00 16200.00 15950.00 15100.00 17300.00 14350.00 15700.00;
    27750.00 27300.00 27050.00 27400.00 27250.00 26350.00 25500.00 25600.00 26250.00 27900.00 25250.00 24700.00;
    49100.00 51200.00 51050.00 53000.00 54250.00 56700.00 57650.00 57250.00 54500.00 66700.00 62900.00 59300.00;
    17500.00 17100.00 17050.00 16650.00 15950.00 16000.00 15550.00 16100.00 17100.00 13500.00 17000.00 14450.00;
    21700.00 22150.00 21900.00 20900.00 21000.00 20800.00 20200.00 20600.00 18450.00 20750.00 18150.00 20050.00;
    44800.00 46000.00 46150.00 47450.00 48800.00 50400.00 50650.00 51700.00 54050.00 59050.00 50500.00 57450.00;
    18350.00 18050.00 18550.00 17950.00 17650.00 17700.00 17450.00 16650.00 16050.00 16250.00 17100.00 15350.00;
    28750.00 28300.00 28700.00 27450.00 27000.00 27000.00 26550.00 26000.00 28200.00 27700.00 22550.00 26900.00;
    56150.00 57300.00 58400.00 57750.00 59950.00 60450.00 62300.00 62600.00 72350.00 70250.00 66350.00 75850.00;
    14150.00 14050.00 13500.00 13400.00 13700.00 13250.00 13050.00 13000.00 12600.00 13500.00 13350.00 11200.00;
    22050.00 21850.00 21100.00 20650.00 21200.00 20250.00 20350.00 20100.00 20000.00 21450.00 19750.00 18250.00;
    43850.00 44000.00 45100.00 47050.00 46600.00 48450.00 50700.00 51700.00 47200.00 57550.00 59450.00 51400.00;
    19650.00 19300.00 18950.00 19050.00 18500.00 17650.00 18150.00 17400.00 19350.00 15200.00 16700.00 16700.00;
    27350.00 26350.00 26550.00 25700.00 25650.00 25850.00 25450.00 24100.00 22900.00 24800.00 22400.00 23750.00;
    50750.00 50850.00 52200.00 55550.00 56800.00 57050.00 59000.00 59550.00 54250.00 67050.00 69200.00 62100.00;
    17400.00 17050.00 17450.00 17300.00 16650.00 16600.00 16000.00 16400.00 18000.00 17300.00 15800.00 16050.00;
    23100.00 22900.00 22550.00 21650.00 21950.00 20900.00 21450.00 21150.00 21650.00 19150.00 18850.00 18100.00;
    49150.00 51700.00 53150.00 52550.00 54650.00 55200.00 57850.00 57850.00 54500.00 56550.00 58950.00 60800.00;
    21800.00 21100.00 21100.00 20750.00 20350.00 20250.00 19650.00 19850.00 20750.00 20100.00 19800.00 17100.00;
    30850.00 30000.00 29550.00 29950.00 29400.00 28350.00 28000.00 27750.00 26000.00 26000.00 28500.00 27400.00;
    53200.00 55150.00 54900.00 56950.00 58600.00 60300.00 61750.00 62500.00 59300.00 71300.00 61850.00 66650.00
]

# Conditional whole-lot arrival hazards by shipment age.
hazard = Dict(
    2 => 0.40,
    3 => 0.80,
    4 => 1.00,
)

# Initial inventory and backorder state.
Ir_init = [45_000.0]
Br_init = [0.0]

# Shipments already in transit at the beginning of the simulation.
shipments_in_transit_init = Shipment[]
push!(shipments_in_transit_init, Shipment(1, 0, 12_000.0))
push!(shipments_in_transit_init, Shipment(1, -1, 8_000.0))


# ------------------------------------------------------------------------------
# Experiment definitions
# ------------------------------------------------------------------------------

function main(; experiment_type::Symbol = :diagnostic)
    n_sims = 400
    eval_start = 6
    eval_end = 18

    baseline = PolicyConfig("Baseline", 0.0, 0.0, :expected)
    safety_stock_10k = PolicyConfig("Safety Stock (10k)", 10_000.0, 0.0, :expected)
    demand_uplift_10pct = PolicyConfig("Demand Uplift (10%)", 0.0, 0.10, :expected)
    conservative_lt = PolicyConfig("Conservative Lead Time", 0.0, 0.0, :worst_case)

    if experiment_type == :diagnostic
        for policy in [baseline, safety_stock_10k, demand_uplift_10pct, conservative_lt]
            run_diagnostic(
                policy;
                forecast_versions = forecast_versions_2024_2025,
                hazard = hazard,
                Ir_init = Ir_init,
                Br_init = Br_init,
                shipments_in_transit_init = shipments_in_transit_init,
                eval_start = eval_start,
                eval_end = eval_end,
            )
        end

    elseif experiment_type == :safety_stock
        policies = [
            PolicyConfig("SS = $(Int(ss))", ss, 0.0, :expected)
            for ss in [0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0,
                       25_000.0, 30_000.0, 40_000.0, 50_000.0, 80_000.0]
        ]

        summary = evaluate_policies(
            policies;
            forecast_versions = forecast_versions_2024_2025,
            hazard = hazard,
            Ir_init = Ir_init,
            Br_init = Br_init,
            shipments_in_transit_init = shipments_in_transit_init,
            n_sims = n_sims,
            eval_start = eval_start,
            eval_end = eval_end,
        )
        print_monte_carlo_summary(summary)

    elseif experiment_type == :demand_uplift
        policies = [
            PolicyConfig("Uplift = $(Int(round(alpha * 100)))%", 0.0, alpha, :expected)
            for alpha in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        ]

        summary = evaluate_policies(
            policies;
            forecast_versions = forecast_versions_2024_2025,
            hazard = hazard,
            Ir_init = Ir_init,
            Br_init = Br_init,
            shipments_in_transit_init = shipments_in_transit_init,
            n_sims = n_sims,
            eval_start = eval_start,
            eval_end = eval_end,
        )
        print_monte_carlo_summary(summary)

    elseif experiment_type == :worst_case
        low_shortage_params = SimulationParams(32.0, 0.025, 1.0, 1.5, -0.30, 0.30, 1)

        summary = evaluate_policies(
            [conservative_lt];
            forecast_versions = forecast_versions_2024_2025,
            hazard = hazard,
            Ir_init = Ir_init,
            Br_init = Br_init,
            shipments_in_transit_init = shipments_in_transit_init,
            #params = low_shortage_params,
            n_sims = n_sims,
            eval_start = eval_start,
            eval_end = eval_end,
        )
        print_monte_carlo_summary(summary)

    else
        error("Unknown experiment_type=$experiment_type. Use :diagnostic, :safety_stock, :demand_uplift, or :worst_case.")
    end
end

# Change this value to select the experiment to run.
# Options: :diagnostic, :safety_stock, :demand_uplift, :worst_case

main(experiment_type = :diagnostic)

