### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 472c37a0-c539-11eb-1f98-cf82dd2c3ddf
begin 
	using LinearAlgebra 
	using Yao 
	using YaoPlots 
	using YaoExtensions
end

# ╔═╡ f634931c-c3dc-443d-9bbe-2bb95a351372
# To build the quantum fourier transform. 

function qft_naive(n)
	circuit = chain(n);
	for i=1:n 
		push!(circuit,put(i=>H));
		if(i!=n)
			for j=i+1:n 
				k = j-i; 
				k = k+1; 
				push!(circuit,control(j,i=>label(Rz(2*pi*1im/2^k),"Rk")));
			end
		end
	end
	s_circuit = circuit; 
	for i=1:floor(n/2)
		s_circuit = push!(circuit,swap(Int(i),Int(n-(i-1))))
	end
	return s_circuit
end

# ╔═╡ 96cb8a51-efc7-49bd-9851-7fdb01764519
# We test my implementation against the standard api for validation 
begin 
	yao_qft = qft(3)
	myqft = qft_naive(3)
	initial = rand_state(3); 
	result = initial |> myqft; 
	reference = initial |> yao_qft;
	@assert(state(reference)≈state(result));
end

# ╔═╡ b89e2a6a-f78c-4c90-9ca3-e5630e2bd6d1
vizcircuit(myqft')

# ╔═╡ d9ec08a5-168f-4a68-aa66-f91d525a2199


# ╔═╡ Cell order:
# ╠═472c37a0-c539-11eb-1f98-cf82dd2c3ddf
# ╠═f634931c-c3dc-443d-9bbe-2bb95a351372
# ╠═96cb8a51-efc7-49bd-9851-7fdb01764519
# ╠═b89e2a6a-f78c-4c90-9ca3-e5630e2bd6d1
# ╠═d9ec08a5-168f-4a68-aa66-f91d525a2199
