### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 9debb076-8a13-4c68-809a-38f0015d3c80
begin
	using LinearAlgebra
	using Yao
	using YaoPlots
	using YaoExtensions
end

# ╔═╡ 0131a822-5938-4cb7-89e6-ad8b58f19a66
begin
	using StatsBase: Histogram, fit
	using Plots: bar, scatter!, gr; gr()
	using BitBasis
	function plotmeasure(x::Array{BitStr{n,Int},1}) where n
		hist = fit(Histogram, Int.(x), 0:2^n)
		x = 0
		if(n<=3)
			s=8
		elseif(n>3 && n<=6)
			s=5
		elseif(n>6 && n<=10)
			s=3.2
		elseif(n>10 && n<=15)
			s=2
		elseif(n>15)
			s=1
		end
		bar(hist.edges[1] .- 0.5, hist.weights, legend=:none, size=(550*(2^n)/s,700), ylims=(0:maximum(hist.weights)), xlims=(0:2^n), grid=:true, ticks=false, border=:none, color=:lightblue, lc=:lightblue)
		scatter!(0:2^n-1, ones(2^n,1), markersize=0,
         series_annotations="|" .* string.(hist.edges[1]; base=2, pad=n) .* "⟩")
		scatter!(0:2^n-1, zeros(2^n,1) .+ maximum(hist.weights), markersize=2,
         series_annotations=string.(hist.weights))
	end
end

# ╔═╡ d9e3af2b-66d5-40a4-88a3-7f61db54a845
# We build today a quantum phase estimation circuit.
# The steps that follow are documented from the lecture "Einfuehrung in der Quantum Computing" at SCCS Informatics 5.

# ╔═╡ d167d7c8-5b14-4c1d-8f62-f83db7245e91
begin
	"""
		Quantum Phase estimation is an important algorithm that is useful to compute the inverse of the eigenvalues in several applications.

	Input : neig::Int64 is the number of qubits used in the approximation of the eigenvalues.

	"""

	function PE(neig,nvec,U)
		uniform_superposition = chain(neig+nvec,repeat(H,1:neig));
		quantum_simulation = chain(neig+nvec);
		U = GeneralMatrixBlock(U)
		for i=1:neig
			push!(quantum_simulation,control(neig+nvec,(i,),[neig+1:(neig+nvec)...,]=>label(U,"U^(2^n)")));
			if(i!=neig)
				U = U*U;
			end
		end
		IQFT = subroutine(neig+nvec,qft(neig),[1:neig...,]);
		circuit = chain(uniform_superposition,quantum_simulation,IQFT)
		return circuit
	end
end

# ╔═╡ 66861e2a-2068-4d51-8645-791e5b213f81

begin
	R = rand(2,2);
	A,B = qr(R);
	circuit = PE(5,1,A)'; 
	# vizcircuit(circuit)
end

# ╔═╡ 8368cdbb-50b7-41dd-a465-64a56fc5b3d2
let
	initial = zero_state(6); 
	measured = (initial |> circuit) |> r->measure(r,nshots=1024);
	plotmeasure(measured)
	# measured
end

# ╔═╡ 92dc1dcb-cedf-4836-a65a-c67883608ed4
let
R = rand(8,8);
A,B = qr(R);
E = eigen(A)
end

# ╔═╡ feeaf2b9-dda1-4167-8734-086613e887ec
begin 
	x = 0.2
	exp(2*pi*1im*x)
end

# ╔═╡ Cell order:
# ╠═9debb076-8a13-4c68-809a-38f0015d3c80
# ╠═d9e3af2b-66d5-40a4-88a3-7f61db54a845
# ╠═0131a822-5938-4cb7-89e6-ad8b58f19a66
# ╠═d167d7c8-5b14-4c1d-8f62-f83db7245e91
# ╠═66861e2a-2068-4d51-8645-791e5b213f81
# ╠═8368cdbb-50b7-41dd-a465-64a56fc5b3d2
# ╠═92dc1dcb-cedf-4836-a65a-c67883608ed4
# ╠═feeaf2b9-dda1-4167-8734-086613e887ec
