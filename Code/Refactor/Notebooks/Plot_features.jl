### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 019df11a-f471-479d-8516-28b21c5cb12a
begin
	using Yao 
	using YaoPlots 
end

# ╔═╡ 279f96f7-1600-46b0-909a-2238c782b062
let
	number_of_qubits = 5; 
	circuit = chain(5,put(1=>X),put(2=>X),put(3=>H));
	plot(circuit)
end

# ╔═╡ 3ef9b79b-564f-47b4-8335-9d25205715ca
let 
	n = 5; 
	circuit2 = chain(n,repeat(H,1:5)); 
	plot(circuit2)
end

# ╔═╡ 97e8be43-3226-4332-a05c-6163cc3b6579
let 
	circuit = chain(3,put(1=>Y),put(1=>X),put(1=>H),repeat(Z,1:3),repeat(X,1:3))
	plot(circuit)
end

# ╔═╡ fe6e1026-f9ee-4e92-94df-765f6b9adf52
let 
	plot(chain(2,control(1,2=>X)))
end

# ╔═╡ bdeaebce-22f1-466b-8da0-b3f100aab840
begin
	bellcircuit = chain(2,put(1=>H),control(1,2=>X))
	s = zero_state(2); 
	res = s |> bellcircuit
	state(res)
end

# ╔═╡ 129d961e-042a-4a38-86b1-305eaf1c89be
q1 = ArrayReg(bit"11")

# ╔═╡ 84f504f4-debc-468a-bdcb-6850a7227889
# TO build a bell state

a = ArrayReg(bit"00") + ArrayReg(bit"11") |> normalize!

# ╔═╡ e2b0edcb-06e3-4fa9-8435-7ccb7d69cd98
begin
	stat = ArrayReg(bit"010") |> normalize! 
	reciprocal1 = chain(3,control(3,1=>Ry(pi)),control(2,1=>Ry(pi)))
	plot(reciprocal1)
	state(stat |> reciprocal1)
end

# ╔═╡ b1c6c3fc-3bc9-4536-a1b6-848c611dfb26
begin 
	Measuregate = chain(2,repeat(H,1:2),Measure(2,locs=1:2));
	plot(Measuregate)
	zero_state(2) |> Measuregate |> r->measure(r,nshots=1024)
end

# ╔═╡ 68afac25-72bf-42d2-806c-8fc10c91648b


# ╔═╡ Cell order:
# ╠═019df11a-f471-479d-8516-28b21c5cb12a
# ╠═279f96f7-1600-46b0-909a-2238c782b062
# ╠═3ef9b79b-564f-47b4-8335-9d25205715ca
# ╠═97e8be43-3226-4332-a05c-6163cc3b6579
# ╠═fe6e1026-f9ee-4e92-94df-765f6b9adf52
# ╠═bdeaebce-22f1-466b-8da0-b3f100aab840
# ╠═129d961e-042a-4a38-86b1-305eaf1c89be
# ╠═84f504f4-debc-468a-bdcb-6850a7227889
# ╠═e2b0edcb-06e3-4fa9-8435-7ccb7d69cd98
# ╠═b1c6c3fc-3bc9-4536-a1b6-848c611dfb26
# ╠═68afac25-72bf-42d2-806c-8fc10c91648b
