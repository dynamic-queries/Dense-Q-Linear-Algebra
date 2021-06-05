using LinearAlgebra
using Yao
using YaoExtensions
using Plots
using BitBasis
#Discretization density
function discretization_density(n::Int)
    return 2^n+2;
end

# Laplacian
function laplacian(n::Int)
    k_m = -2*ones(n);
    k_ad = 1*ones(n-1);
    return Tridiagonal(k_ad,k_m,k_ad);
end

function Qsim(A)
    # We simulate for t = 2π;
    return exp(2π*1im*Matrix(A));
end
# Dicretize
function discretize(x_initial::Float64,x_final::Float64,N::Int)
    x = LinRange(x_initial,x_final,N);
    return x;
end

# Normalize
function normalize(x::Vector{Float64})
    return x./norm(x);
end

# Initial State
function IC(x::LinRange{Float64},f,BC,dt)
    n = length(x);
    z = f(x);
    h = x[2]-x[1];
    y = z[2:n-1]#*(h^2/dt);
    y[1] = y[1] + BC[1];
    y[end] = y[end] + BC[2];
    z = normalize(z);
    y = Vector(normalize(y));
    return z,Vector{ComplexF64}(y);
end

# Phase estimation
function Phase_estimation(U,nr,nq)
    n_tot = nr + nq;
    n_tot = Int(n_tot);
    U = GeneralMatrixBlock(U);
    # Uniform Superposition
    # us_c = chain(n_tot, put(i=>H) for i in 1:nr);
    us_c = repeat(n_tot,H,1:nr);
    # Controlled Simulations
    cr_c  = chain(n_tot);
    for k = 1:nr
        push!(cr_c,control(n_tot,(k,),(nr+1:n_tot...,)=>U));
        if(k!=nr)
            U = matblock(mat(U)*mat(U));
        end
    end
    # Chained together
    circuit = chain(us_c,cr_c,subroutine(n_tot,qft(nr)',[1:nr...,]));
    circuit;
end

# Inverse_phase_estimation
function Inverse_phase_estimation(U,nr,nq)
    return Phase_estimation(U,nr,nq)';
end

# controlled_rotation # TODO
function controlled_rotation()

end

# Eigenvalue inversion
# Doing this classically is far efficient
# Nevertheless, we provide a quantum implementation. However the code is not refactored.
function invert_register(reg::ArrayReg)  # TODO

end

#check state
function check_state(A,b,U);
    @assert(A≈A');
    @assert(norm(b)≈1);
    @assert(U*U'≈I);
end

# Size of the matrix : nqubits
function compute_q(A)
    n = size(A)[1];
    return log(2,n);
end

# API from Yao
struct HHLCRot{N, NC, T} <: PrimitiveBlock{N}
    cbits::Vector{Int}
    ibit::Int
    C_value::T
    HHLCRot{N}(cbits::Vector{Int}, ibit::Int, C_value::T) where {N, T} = new{N, length(cbits), T}(cbits, ibit, C_value)
end

@inline function hhlrotmat(λ::Real, C_value::Real)
    b = C_value/λ
    display(b);
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
            #display(λ)
            if λ >= hr.C_value
                u = hhlrotmat(λ, hr.C_value)
                YaoArrayRegister.u1rows!(state(reg), i, i+step, u...)
            end
        end
    end
    reg
end


# HHL solver
function solve(A,b,n)
    # A : Hermitian matrix
    # b : valid qunatum state
    # n : Number of registers for eigenvalues
    # Phase estimation
    U = Qsim(A);
    check_state(A,b,U);
    nq = compute_q(A);
    n_tot = nq + n;
    pe = Phase_estimation(U,n,nq);
    A = Matrix(A);
    C_value = minimum(eigvals(A) .|> abs)*0.25;
    #display( C_value);
    cr = HHLCRot{n+1}([2:n+1...,], 1, C_value);
    ipe = pe';
    n_cum = Int(1 + n_tot);
    circuit = chain(n_cum,subroutine(n_cum,pe,[2:n_cum...,]),subroutine(n_cum,cr,[1:(n+1)...,]),subroutine(n_cum,ipe,[2:n_cum...,]));
    ψ = join(ArrayReg(b),zero_state(n),zero_state(1));
    apply!(ψ,circuit);
    #hhlproject!(ψ,n)./C_value;  # ψ is a register. Needs projection. Do not measure. Loses all data.
end

function hhlproject!(all_bit::ArrayReg, nr::Int)
    all_bit |> focus!(1:(nr+1)...) |> select!(1) |> state |> vec
end
