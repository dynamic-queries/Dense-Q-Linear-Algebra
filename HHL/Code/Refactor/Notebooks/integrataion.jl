### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ f2214427-46b2-4567-9dd0-36776d3d6871
begin
	using Yao
	using YaoExtensions
	using BitBasis
	using Test, LinearAlgebra
end

# ╔═╡ ce2699f6-b8c6-4649-ae12-ef71d2c5b926
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
    chain(hs, control_circuit, iqft)
end

# ╔═╡ def07fd0-e255-4d93-9feb-4d84b1cce580
begin
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
	
end

# ╔═╡ a8d90462-99bf-47ba-9389-02b59790be8e
function compute_θ(nreg,scale,t,ntot)
	# We estimate angles corresponding to every element of the state vector
	circuit = chain(ntot); 
	l =  2^nreg; 
	angles = zeros(l-1); 
	for i=1:l-1
		val = (scale*l)/i; 
		if (val ≈ 1)
			angles[i] = 2*asin(val); 
		elseif (val < 1) 
			angles[i] = 2*asin(val); 
		else 
			angles[i] = 0 ; 
		end
	# We now have the list of all possible angles that can arise 
	# This means that there should be l possible gate implementations
	control_vec = Vector(2:(nreg+1)); 
	for j=1:(nreg)
		if(readbit(i,j)==0)
			control_vec[j] = -control_vec[j]; 
		end 
	end
	push!(circuit,control(control_vec,1=>Ry(angles[i]/t)));
	end
	return circuit;
end

# ╔═╡ 4f7f5f7e-4606-417a-bef7-1006cb69ecbf
function reciprocal(nreg,scaling)
	ntot = nreg+1
	circuit = chain(ntot);
	for j = 2:ntot
		sub_circuit = compute_θ(nreg,scaling,j-1,ntot);
		circuit = chain(circuit,sub_circuit); 
		nreg = nreg-1; 
	end 	
	return circuit;
end

# ╔═╡ 6dc87559-eba3-4881-bdbf-e0686780cf14
function hhlproject!(all_bit::ArrayReg, n_reg::Int)
    all_bit |> focus!(1:(n_reg+1)...) |> select!(1) |> state |> vec
end

# ╔═╡ a6d70eb8-5cad-4094-a56d-041e9f1b2cf9
function hhlcircuit(UG, n_reg::Int, C_value::Real)
    n_b = nqubits(UG)
    n_all = 1 + n_reg + n_b
    pe = PEBlock(UG, n_reg, n_b)
    cr1 = HHLCRot{n_reg+1}([2:n_reg+1...,], 1, C_value) #
    chain(n_all, subroutine(n_all, pe, [2:n_all...,]), subroutine(n_all, cr1
			, [1:(n_reg+1)...,]), subroutine(n_all, pe', [2:n_all...,]))
end

# ╔═╡ 9471c03e-93ba-4ec9-94aa-46d05fb5f5b3
function hhlsolve(A::Matrix, b::Vector, n_reg::Int, C_value::Real)
    if !ishermitian(A)
        throw(ArgumentError("Input matrix not hermitian!"))
    end
    UG = matblock(exp(2π*im.*A))

    # Generating input bits
    all_bit =  join(ArrayReg(b), zero_state(n_reg), zero_state(1))

    # Construct HHL circuit./
    circuit = hhlcircuit(UG, n_reg, C_value)

    # Apply bits to the circuit.
    all_bit = all_bit |> circuit

    # Get state of aiming state |1>|00>|u>.
    hhlproject!(all_bit, n_reg) ./ C_value
end

# ╔═╡ 8f4cb10f-4dad-4603-83ea-3c0403d0b21f
function hhl_problem(nbit::Int)
    siz = 1<<nbit
	A = Matrix(Tridiagonal(-1*ones(siz-1),2*ones(siz),-1*ones(siz-1)));
	@assert(A'*A ≈ A'*A)
	E = Matrix(1.0I,siz,siz);
	# A = kron(A,E) + kron(E,A);
	# A = kron(A,E,E) + kron(E,A,E) + kron(E,E,A);  
    # b = normalize(ones(ComplexF64,siz*siz))
	b = normalize(ones(ComplexF64,siz))
	# b = normalize(ones(ComplexF64,siz*siz*siz)); 

    Matrix(A), b
end

# ╔═╡ c6a51151-19a1-4b72-8cc8-a132ad51928f
begin
    # Set up initial conditions.
    ## A: Matrix in linear equation A|x> = |b>.
    ## signs: Diagonal Matrix of eigen values of A.
    ## base_space: the eigen space of A.
    ## x: |x>.
    using Random
    Random.seed!(2)
    N = 2
    A, b = hhl_problem(N)
    x = A^(-1)*b # base_i = base_space[:,i] ϕ1 = (A*base_i./base_i)[1]

    ## n_b  : number of bits for |b>.
    ## n_reg: number of PE register.
    ## n_all: number of all bits.
	Error = zeros(12);
	for n_reg = 1:12
		## C_value: value of constant C in control rotation.
		## It should be samller than the minimum eigen value of A.
		C_value = minimum(eigvals(A) .|> abs)*0.25
		# C_value = 1.0/(1<<n_reg) * 0.9
		res = hhlsolve(A, b, n_reg, C_value)
		Error[n_reg] = sqrt(norm(res-x));
	end
	using Plots
	Plots.plot(1:12,Error,xlabel=:"n_eig",ylabel="||ϵ||",label="N=4")
	# savefig("Error_analysis")
	# Plots.plot([r.re for r in res],label=:"HHL - 16IP");
	# Plots.plot!([xo.re for xo in x],label="True");
	# savefig("3D - 16")
end

# ╔═╡ f613c35e-6cf0-46db-9fea-a5f3ceb301a3
let
	N = 5;
	A, b = hhl_problem(N)
	# plot!(eigen(A).values,label="N=32",ylabel=:"λ")
	savefig("Eigenvalues")
end

# ╔═╡ Cell order:
# ╠═f2214427-46b2-4567-9dd0-36776d3d6871
# ╠═ce2699f6-b8c6-4649-ae12-ef71d2c5b926
# ╠═def07fd0-e255-4d93-9feb-4d84b1cce580
# ╠═a8d90462-99bf-47ba-9389-02b59790be8e
# ╠═4f7f5f7e-4606-417a-bef7-1006cb69ecbf
# ╠═6dc87559-eba3-4881-bdbf-e0686780cf14
# ╠═a6d70eb8-5cad-4094-a56d-041e9f1b2cf9
# ╠═9471c03e-93ba-4ec9-94aa-46d05fb5f5b3
# ╠═8f4cb10f-4dad-4603-83ea-3c0403d0b21f
# ╠═c6a51151-19a1-4b72-8cc8-a132ad51928f
# ╠═f613c35e-6cf0-46db-9fea-a5f3ceb301a3
