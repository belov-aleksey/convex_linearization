using Plots
using Optim

function calc_p_q(dfdx::Vector, x0::Vector)
    p = []
    q = []
    for i in eachindex(dfdx)
        push!(p, maximum([dfdx[i](x0), 0]))
        push!(q, maximum([-dfdx[i](x0)*x0[i]^2, 0]))
    end
    return p, q
end

function A_(p_f, p_g)
    A = []
    for i in eachindex(p_f)
        push!(A, λ -> p_f[i]+λ*p_g[i])
    end
    return A
end

function B_(q_f, q_g)
    B = []
    for i in eachindex(q_f)
        push!(B, λ -> q_f[i]+λ*q_g[i])
    end
    return B
end

function dLdx(A, B)
    res = []
    for i in eachindex(A)
        push!(res, (x, λ) -> A[i](λ)-B[i](λ)/x^2)
    end
    return res
end

function x_opt(λ, A, B, x_range, DLdx)
    x = []
    for i in eachindex(DLdx)
        alpha = x_range[i][1]
        beta = x_range[i][2]
        if DLdx[i](beta, λ) <= 0
            xi = beta
        elseif DLdx[i](alpha, λ) >= 0
            xi = alpha
        else
            xi = sqrt(B[i](λ)/A[i](λ))
        end
        push!(x, xi)
    end
    return x
end

function conlin_approximate(f, dfdx::Vector, x0)
    p, q = calc_p_q(dfdx, x0)

    c = f(x0)
    for i in eachindex(x0)
        c -= p[i]*x0[i] + q[i]/x0[i]
    end
    function fc(x)
        res = c
        for i in eachindex(x)
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
    x_range,
)
    # CONLIN-аппроксимация целевой функции и функции ограничений
    gc = conlin_approximate(g, dgdx, x0)
    Vc = conlin_approximate(V, dVdx, x0)

    # pi, pji, qi, qji
    p_v, q_v = calc_p_q(dVdx, x0)
    p_g, q_g = calc_p_q(dgdx, x0)

    # Ai, Bi
    A = A_(p_v, p_g)
    B = B_(q_v, q_g)

    # Градиент Лагранжиана
    dldx = dLdx(A, B)

    # Функция x(λ)
    x(λ) = x_opt(λ, A, B, x_range,dldx)
    # Функция ϕ(λ)
    ϕ(λ) = Vc(x(λ)) + λ*gc(x(λ))

    # Для построения графика 
    # λ_vector=range(0, stop=30,length=100)
    # ϕ_vector=[ϕ(λ) for λ in λ_vector]
    
    # Функция -ϕ (метод золотого сечения ищет только минимум, нужен максимум)
    ϕ_(λ) = -(Vc(x(λ)) + λ*gc(x(λ)))
    λ_opt_iter = optimize(ϕ_, 0.0, 100.0, GoldenSection())

    # Проектные переменные
    x_new = x(λ_opt_iter.minimizer)
    return x_new
end

norm(vector) = sqrt(sum(vector.^2))
function conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    x_range,    
)
    x = conlin_iter(V, dVdx, g, dgdx, x0, x_range)
    i = 1
    while norm(abs.(x-x0))>=0.01 && i < 100
        println("x=$x")
        x0 = x
        x = conlin_iter(V, dVdx, g, dgdx, x, x_range)
    end
end

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
x_range = [x1_range, x2_range]
x0=[0.5,2.4]

conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    x_range    
)