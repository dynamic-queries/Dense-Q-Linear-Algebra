
# Parameters to fiddle with
#====================================================#
iter = 1;
n_reg = 10;
#====================================================#
A,b,z  = hhl_problem();
# Julia iterative solvers
x_actual = A^(-iter)*b;
C_val = minimum(eigvals(A) .|> abs)*0.25
res = b;
res = hhlsolve(A,res,n_reg,C_val);
plot(z[2:length(z)-1],real(res),color="blue",label="HHL",title=:"Steady state equation u_{xx} = f(x)");
plot!(z[2:length(z)-1],real(x_actual),color="red",label="Actual");
