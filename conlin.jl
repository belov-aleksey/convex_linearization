using Plots
using Optim

# Целевая функция
V(x)= x[1] + x[2]
dVdx1(x) = 1
dVdx2(x) = 1
dVdx = [dVdx1, dVdx2]

# Функция ограничений
g(x) = 8/(16*x[1] + 9*x[2]) - 4.5/(9*x[1] + 16*x[2]) - 0.1
dgdx1(x) = -128/(16*x[1]+9*x[2])^2+40.5/(9*x[1]+16*x[2])^2
dgdx2(x) = -72/(16*x[1]+9*x[2])^2+72/(9*x[1]+16*x[2])^2
dgdx = [dgdx1, dgdx2]

x1_range=[0.2,2.5]
x2_range=[0.2,2.5]
x0=[2,1]
alpha=0.2
beta=2.5

function calc_p_q(f, dfdx::Vector, x0::Vector)
    p = []
    q = []
    for i in 1:length(dfdx)
        push!(p, maximum([dfdx[i](x0), 0]))
        push!(q, maximum([-dfdx[i](x0)*x0[i]^2, 0]))
    end
    return p, q
end
pi(dfdxi, x0::Vector) =  maximum([dfdxi(x0), 0])
qi(dfdxi, x0::Vector, xi) =  maximum([-dfdxi(x0)*xi^2, 0])

function create_A(p_f, p_g)
    A = []
    for i in 1:length(p_f)
        push!(A, lam -> p_f[i]+lam*p_g[i])
    end
    return A
end
Ai(p_fi, p_gi) = lam -> p_fi+lam*p_gi

function create_B(q_f, q_g)
    B = []
    for i in 1:length(q_f)
        push!(B, lam -> q_f[i]+lam*q_g[i])
    end
    return B
end
Bi(q_fi, q_gi) = lam -> q_fi+lam*q_gi

function dLdx(A, B)
    res = []
    for i in 1:length(A)
        push!(res, (x, lam) -> A[i](lam)-B[i](lam)/x^2)
    end
    return res
end
dLdxi(A,B) = (x, lam) -> A(lam)-B(lam)/x^2

function x_opt(lam, A, B, x1_range, x2_range)
    x = []
    gradL = dLdx(A,b)
    for i in 1:length(A)
        push!(x, lam -> sqrt(A[i](lam)/B[i](lam)))
    end
    return x
end

function xi_opt(lam, Ai, Bi, alpha, beta, DLdxi)
    if DLdxi(beta, lam) <= 0
        xi = beta
    elseif DLdxi(alpha, lam) >= 0
        xi = alpha
    else
        sqrt(Bi(lam)/Ai(lam))
    end
end

function conlin_approximate(f, dfdx::Vector, x0)
    p = []
    q = []
    for i in 1:2
        push!(p, maximum([dfdx[i](x0), 0]))
        push!(q, maximum([-dfdx[i](x0)*x0[i]^2, 0]))
    end
    c = f(x0)
    for i in 1:2
        c -= p[i]*x0[i] + q[i]/x0[i]
    end
    function fc(x)
        res = c
        for i in 1:2
            res += p[i]*x[i] + q[i]/x[i]
        end
        return res
    end
    return fc
end

function conlin_iter(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    alpha,
    beta
)
    # CONLIN-аппроксимация целевой функции и функции ограничений
    gc = conlin_approximate(g, dgdx, x0)
    Vc = conlin_approximate(V, dVdx, x0)

    # pi, pji, qi, qji
    p_v1=pi(dVdx1,x0)
    p_v2=pi(dVdx2,x0)
    q_v1=qi(dVdx1,x0,x0[1])
    q_v2=qi(dVdx2,x0,x0[2])

    p_g1=pi(dgdx1,x0)
    p_g2=pi(dgdx2,x0)
    q_g1=qi(dgdx1,x0,x0[1])
    q_g2=qi(dgdx2,x0,x0[2])

    # Ai, Bi
    A1=Ai(p_v1, p_g1)
    A2=Ai(p_v2, p_g2)
    B1=Bi(q_v1, q_g1)
    B2=Bi(q_v2, q_g2)

    # Градиент Лагранжиана
    dldx1 = dLdxi(A1,B1)
    dldx2 = dLdxi(A2,B2)

    # Функция phi(lambda)
    phi(lam) = Vc([xi_opt(lam,A1,B1,alpha,beta,dldx1), xi_opt(lam,A2,B2,alpha,beta,dldx2)])+lam*gc([xi_opt(lam,A1,B1,alpha,beta,dldx1),xi_opt(lam,A2,B2,alpha,beta,dldx2)])
    lambda_vector=range(0, stop=30,length=100)
    phi_vector=[phi(lam) for lam in lambda_vector]

    # Функция -phi (метод золотого сечения ищет только минимум, нужен максимум)
    phi_(lam) = -(Vc([xi_opt(lam,A1,B1,alpha,beta,dldx1), xi_opt(lam,A2,B2,alpha,beta,dldx2)])+lam*gc([xi_opt(lam,A1,B1,alpha,beta,dldx1),xi_opt(lam,A2,B2,alpha,beta,dldx2)]))
    lambda_opt_iter = optimize(phi_, 0.0, 100.0, GoldenSection())

    # Проектные переменные
    x1_iter = xi_opt(lambda_opt_iter.minimizer,A1,B1,alpha,beta,dldx1)
    x2_iter = xi_opt(lambda_opt_iter.minimizer,A2,B2,alpha,beta,dldx2)
    return [x1_iter, x2_iter]
end

norm(vector) = sqrt(sum(vector.^2))
function conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    alpha,
    beta    
)
    x = conlin_iter(V, dVdx, g, dgdx, x0,alpha, beta)
    i = 1
    while norm(abs.(x-x0))>=0.01 && i < 100
        println("x=$x")
        x0 = x
        x = conlin_iter(V, dVdx, g, dgdx, x,alpha, beta)
    end
end



conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    alpha,
    beta    
)