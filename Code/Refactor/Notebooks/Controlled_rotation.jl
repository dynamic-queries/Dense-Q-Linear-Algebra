### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ d3572f90-c53f-11eb-3fb0-5b5a663e621b
let 
	using Yao 
	using YaoPlots 
	using Plots: plot,plot!
	using LinearAlgebra
	using YaoExtensions
	using BitBasis 
	using StatsBase: Histogram, fit
	using Plots: bar, scatter!, gr; gr()
end

# ╔═╡ 8512eb92-e96d-427d-b3bf-6fe17ab11bfa

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
	bar(hist.edges[1] .- 0.5, hist.weights, legend=:none, size=(500*(2^n)/s,1000), ylims=(0:maximum(hist.weights)), xlims=(0:2^n), grid=:true, ticks=false, border=:none, color=:lightblue, lc=:lightblue)
	scatter!(0:2^n-1, ones(2^n,1), markersize=0,
	 series_annotations="|" .* string.(hist.edges[1]; base=2, pad=n) .* "⟩")
	scatter!(0:2^n-1, zeros(2^n,1) .+ maximum(hist.weights), markersize=2,
	 series_annotations=string.(hist.weights))
end

# ╔═╡ bb839ce9-bb19-4130-a91e-c7d97272af65
function reciprocal_vanilla(nreg)
	circuit = chain(nreg+1); 
	t = pi; 
	for i=2:(nreg+1)
		push!(circuit,control(i,1=>Ry(t)))
		t = t/2; 
	end
	return circuit
end

# ╔═╡ da6ab9c8-12b8-4ae5-9af3-de96360c70ed
vizcircuit(reciprocal_vanilla(5))

# ╔═╡ 33e114a6-13fd-4ea6-aab7-5895eb95c48b
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

# ╔═╡ 9168f16b-65ec-45aa-a4de-0e279212ed0e
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

# ╔═╡ aa52cb6b-133b-4e8e-ac88-0b9086e86b51
begin
	initial = rand_state(4); 
	circuit= reciprocal(3,0.1);
	vizcircuit(circuit);
end

# ╔═╡ df3996ae-ea09-4833-9694-a58d48d5d552
plotmeasure(initial|>r->measure(r,nshots=4*1024))

# ╔═╡ 38611495-8c35-40ef-9d13-3540986345d7
begin
	measured = (initial|>circuit) |> r->measure(r,nshots=4*1024);
	plotmeasure(measured)
end

# ╔═╡ Cell order:
# ╠═d3572f90-c53f-11eb-3fb0-5b5a663e621b
# ╠═8512eb92-e96d-427d-b3bf-6fe17ab11bfa
# ╠═bb839ce9-bb19-4130-a91e-c7d97272af65
# ╠═da6ab9c8-12b8-4ae5-9af3-de96360c70ed
# ╠═33e114a6-13fd-4ea6-aab7-5895eb95c48b
# ╠═9168f16b-65ec-45aa-a4de-0e279212ed0e
# ╠═aa52cb6b-133b-4e8e-ac88-0b9086e86b51
# ╠═df3996ae-ea09-4833-9694-a58d48d5d552
# ╠═38611495-8c35-40ef-9d13-3540986345d7
