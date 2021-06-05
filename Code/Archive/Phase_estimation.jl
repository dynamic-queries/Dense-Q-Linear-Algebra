begin
    using Yao;
    using YaoExtensions;
    using LinearAlgebra
end
#============================================================================#
function discretization_density(n)
    vector_length = 2^n+2;
    return vector_length;
end
#============================================================================#
function discretize(x_intial,x_final,N)
    x = LinRange(x_intial,x_final,N);
    return x;
end
#============================================================================#
function cournants_tau(N)
    h = 1/(N-1);
    tau = 2/h^2;
    return tau;
end
#============================================================================#
function initial_state(x,f,tau)
    n = length(x);
    # f here is the sin function
    # But in the discretization equation Ax = b we do not account boundary values. So N ~= N-2;
    #n_qubits = log(2,n-1);
    h = x[2]-x[1];
    # Compute f(x_i)
    T_0 = f(x[2:n-1]).*(h^2/tau);
    # Add boundary conditions on to b
    temp1 = zeros(n-2);
    temp1[1] = f(x[1]);
    temp1[2] = f(x[n]);
    temp2 = temp1 + T_0;
    T_0 = temp2;
    # Norm of T_0
    T_0 = T_0 / norm(T_0);
    return Vector{ComplexF64}(T_0) ;
end
#============================================================================#
function toeplitz(n,tau,h)
    k1 = 0.1*(1 + 2*tau/h^2)*ones(n);
    k2 = 0.1*(-tau/h^2)*ones(n-1);
    return Tridiagonal(k2,k1,k2)./(tau/h^2);
end
#============================================================================#
function update_matrix(tau,m,n)
    h = 1/(n+1);
    A = toeplitz(n,tau,h);
    return Matrix(A);
end
#============================================================================#
#= TODO : Reimplemet your own version
function PEBlock(UG::GeneralMatrixBlock, n_reg::Int, n_b::Int)
    nbit = n_b + n_reg
    # Apply Hadamard Gate.
    hs = repeat(nbit, H, 1:n_reg)

    # Construct a control circuit.
    control_circuit = chain(nbit)
    for i = 1:n_reg
        push!(control_circuit, control(nbit, (i,), (n_reg+1:nbit...,)=>UG))
        if i != n_reg
            UG = matblock(mat(UG) * mat(UG))
        end
    end

    # Inverse QFT Block.
    iqft = subroutine(nbit, QFTBlock{n_reg}()',[1:n_reg...,])
    chain(hs, control_circuit, iqft);
end
=##===========================================================================#
function QSim_U(A)
    if (!(A≈adjoint(A)))
        throw(ArgumentError("Non Hermitial input to QSim_U"));
    end
    A = Matrix(A);
    return exp(1im*2*π*A);
end
#===========================================================================#
function perform_checks(A,b)
    @assert(norm(b) ≈ 1);
    @assert(A*adjoint(A)≈I);
    println("Simulation Matrix U is unitary\nSource vector b is normalized.\n");
end
#===========================================================================#
#TODO: Reimplement your version
struct HHLCRot{N, NC, T} <: PrimitiveBlock{N}
    cbits::Vector{Int}
    ibit::Int
    C_value::T
    HHLCRot{N}(cbits::Vector{Int}, ibit::Int, C_value::T) where {N, T} = new{N, length(cbits), T}(cbits, ibit, C_value)
end

@inline function hhlrotmat(λ::Real, C_value::Real)
    b = C_value/λ
    a = sqrt(1-b^2)
    a, -b, b, a
end

function YaoBlocks._apply!(reg::ArrayReg, hr::HHLCRot{N, NC, T}) where {N, NC, T}
    mask = bmask(hr.ibit)
    step = 1<<(hr.ibit-1)
    step_2 = step*2
    nbit = nqubits(reg)
    for j = 0:step_2:size(reg.state, 1)-step
        for i = j+1:j+step
            λ = bfloat(readbit(i-1, hr.cbits...), nbits=nbit-1)
            if λ >= hr.C_value
                u = hhlrotmat(λ, hr.C_value)
                YaoArrayRegister.u1rows!(state(reg), i, i+step, u...)
            end
        end
    end
    reg
end
#===========================================================================#
function HHL()
    #generate_problem();
    begin   #TODO : Move this block of code to a struct;
        zmin = 0;
        zmax = pi;
        n_qubits = 5;
        N = discretization_density(5);
        t_end = 5;
        tau =  0.1
        n_time_steps = t_end/tau;
        f_sample(x) = sin.(x);
        z = discretize(zmin,zmax,N);
        rhs = initial_state(z,f_sample,tau);
        A = update_matrix(tau,n_time_steps,N-2);
        println("Parameters initialized")
        U = QSim_U(A);
        perform_checks(U,rhs);
    end
    # Perform phase estimation on state.
    pe = PEBlock(GeneralMatrixBlock(U),n_qubits,n_qubits);
    println(pe);
    # Perform controlled rotation

end
#===========================================================================#
#===========================================================================#
