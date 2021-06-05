include("Utilities.jl")

function assemble(h,dt,N,time_steps)
    N = N-2; # We consider only internal nodes
    κ = dt/h^2;
    A = laplacian(N);
    M = (I - A)#./(dt/h^2);
    M = M^(time_steps);
end

# parameters
let
    precision = 5;
    zmin = 0.0;
    zmax = 1.0π;
    n_qubits = 5;
    t_end = 1;
    dt = 1;
    f(x) = sin.(x);
    BC = [0,0];

    N = discretization_density(n_qubits);
    z = discretize(zmin,zmax,N);
    n_time_steps = t_end / dt;
    h = z[2]-z[1];
    A = assemble(h,dt,N,n_time_steps);
    b_plot,b = IC(z,f,BC,dt);
    sol = solve(A,b,precision);
    # Plots.plot(z,vcat(BC[1],A^(-1)*b,BC[2]))
end
