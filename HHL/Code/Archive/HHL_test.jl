begin
    using Yao
    using YaoExtensions
    include("HHLLlib.jl")
    include("Phase_estimation.jl")
    using LinearAlgebra
    using BitBasis
    using Plots
    using Random
    Random.seed!(1);
end

function hhl_problem()
    zmin = 0;
    zmax = 2*pi;
    n_qubits = 5;
    N = discretization_density(n_qubits);
    t_end = 5;
    tau =  cournants_tau(N);
    n_time_steps = t_end/tau;
    f_sample(x) = sin.(x);
    z = discretize(zmin,zmax,N);
    rhs = initial_state(z,f_sample,tau);
    A = update_matrix(tau,n_time_steps,N-2);
    #display(cond(A));
    println("Parameters initialized")
    U = QSim_U(A);
    perform_checks(U,rhs);
    A,rhs,z
end
