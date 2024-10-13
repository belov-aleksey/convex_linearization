using Plots
using Optim
using ForwardDiff

function calc_p_q(dfdx::Vector, x0::Vector)
    p = Vector{Float64}()
    q = Vector{Float64}()
    for i in eachindex(dfdx)
        push!(p, maximum([dfdx[i](x0), 0]))
        push!(q, maximum([-dfdx[i](x0)*x0[i]^2, 0]))
    end
    return p, q
end

function A_(p_f, p_g)
    # в p_f элементов столько, сколько xi
    # в p_g столько, сколько ограничений, каждый элемент размером с количеством xi
    A = []
    # в A будет столько элементов, сколько переменных xi

    if p_g isa Vector{Vector{Float64}} # g - вектор из ограничений gi
        new_p_g = [p_g[j][i] for i in 1:length(p_g[1]), j in 1:length(p_g)]
        for i in eachindex(p_f)
            push!(A, λ -> p_f[i] + sum(λ.*new_p_g[i]))
        end
    elseif p_g isa Vector{Float64} # g - единственное ограничение
        for i in eachindex(p_f)
            push!(A, λ -> p_f[i]+λ*p_g[i])
        end
    else
        error("Странный тип данных у p_g")
    end
    return A
end

function B_(q_f, q_g)
    B = []
    if q_g isa Vector{Vector{Float64}}
        new_q_g = [q_g[j][i] for i in 1:length(q_g[1]), j in 1:length(q_g)]
        for i in eachindex(q_f)
            push!(B, λ -> q_f[i] + sum(λ.*new_q_g[i]))
        end
    elseif q_g isa Vector{Float64}
        for i in eachindex(q_f)
            push!(B, λ -> q_f[i]+λ*q_g[i])
        end
    else
        error("Странный тип данных у q_g")
    end
    return B
end

function dLdx(A, B)
    # TODO сейчас x и lam - числа. А надо, чтобы были векторами (точнее x можно оставить числом, но lam -вектор теперь)
    res = []
    for i in eachindex(A)
        push!(res, (x, λ) -> A[i](λ)-B[i](λ)/x^2)
    end
    return res
end

function x_opt(λ, A, B, x_range, x0, DLdx)
    nu=0.5
    x = []
    for i in eachindex(DLdx)
        alpha = maximum([x_range[i][1], x0[i]-nu*(x_range[i][2]- x_range[i][1])])
        beta = minimum([x_range[i][2], x0[i]+nu*(x_range[i][2]- x_range[i][1])])
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

# Conlin-аппроксимация функции f с градиентом dfdx в окрестности точки x0
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

const koeff = 2.0
pow(x,y) = x ^ y
function G_eff(g, koeff)
    if (g >= 0)
        return (1.0 - 1.0/pow((1.0 + pow(g,(1/koeff))),koeff))
    else
        return (1.0/pow((1.0 + pow((-g),(1/koeff))),koeff) - 1.0)
    end
end


function conlin_iter(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    x_range,
)
    if g isa Vector && length(g) != length(dgdx)
        error("Длины векторов g и dgdx не совпадают")
    end

    # CONLIN-аппроксимация целевой функции и функции ограничений
    if g isa Vector
        gc = []
        for i in 1:length(g)
            push!(gc, conlin_approximate(g[i], dgdx[i], x0))
        end
    else
        gc = conlin_approximate(g, dgdx, x0)
    end
    Vc = conlin_approximate(V, dVdx, x0)

    # pi, pji, qi, qji
    p_v, q_v = calc_p_q(dVdx, x0)
    if g isa Vector
        p_g = Vector{Vector{Float64}}()
        q_g = Vector{Vector{Float64}}()
        for i in 1:length(dgdx)
            p_g_i, q_g_i = calc_p_q(dgdx[i], x0)
            push!(p_g, p_g_i)
            push!(q_g, q_g_i)
        end
    else
        p_g, q_g = calc_p_q(dgdx, x0)
    end

    # # Ai, Bi
    A = A_(p_v, p_g)
    B = B_(q_v, q_g)

    # # Градиент Лагранжиана
    dldx = dLdx(A, B)
    # dldx[1](1, [2,2,2,2,2]) # пример вызова если λ - вектор

    # # Функция x(λ)
    x(λ) = x_opt(λ, A, B, x_range, x0, dldx)
    # x([1,2,3,4,5])

    # println(gc[5]([1,2]))
    # # Функция ϕ(λ)
    if gc isa Vector
        ϕ = λ -> Vc(x(λ)) + sum(λ[i] * gc[i](x(λ)) for i in 1:length(gc))
    else
        ϕ = λ -> Vc(x(λ)) + λ*gc(x(λ))
    end
    
    # способ из статьи
    # λ_0 = [0.0 for i in eachindex(gc)] # начальное приближение
    # λ = λ_0
    # gc_values = [gc[i](x(λ)) for i in eachindex(gc)]
    # g_max = maximum(gc_values)
    # r = 0
    # while g_max > eps() && r < 100
    #     g_active = []
    #     for i in eachindex(gc_values)
    #         if gc[i](x(λ)) > -eps()
    #             push!(g_active, gc_values[i])
    #         end
    #     end
    #     g_norm = norm(g_active)
    #     theta = 0.9
    #     for i in eachindex(gc_values)
    #         λ[i] = maximum([0, theta*gc_values[i]/g_norm])
    #     end
    #     gc_values = [gc[i](x(λ)) for i in eachindex(gc)]
    #     g_max = maximum(gc_values)
    #     λ_0= λ
    #     r += 1 
    # end
    # x_new = x(λ)

    # способ с G_eff
    λ_0 = [0.0 for i in eachindex(gc)] # начальное приближение
    λ = λ_0
    gc_values = [gc[i](x(λ)) for i in eachindex(gc)]
    g_max = maximum(gc_values)
    r = 0
    while g_max > 0.0 && r < 100
        for i in eachindex(gc_values)
            λ[i] = λ_0[i] * (1.0 + G_eff(gc_values[i], koeff))
        end
        gc_values = [gc[i](x(λ)) for i in eachindex(gc)]
        g_max = maximum(gc_values)
        λ_0= λ
        r += 1 
    end
    x_new = x(λ)

    # # Для построения графика 
    # # λ_vector=range(0, stop=30,length=100)
    # # ϕ_vector=[ϕ(λ) for λ in λ_vector]
    
    # # Функция -ϕ (метод золотого сечения ищет только минимум, нужен максимум)
    # ϕ_(λ) = -ϕ(λ)
    # λ_opt_iter = Optim.optimize(ϕ_, 0.0, 100.0, GoldenSection())
    # x_new = x(λ_opt_iter.minimizer)

    # lower = [0.0 for i in eachindex(gc)]
    # upper = [100000.0 for i in eachindex(gc)]
    # initial = [10.0 for i in eachindex(gc)]
    # inner_optimizer = GradientDescent()
    # dϕdλ(λ) = ForwardDiff.gradient(ϕ, λ)
    # λ_opt_iter = optimize(ϕ, dϕdλ, lower, upper, initial, Fminbox(GradientDescent()); inplace = false)
    # x_new = x(λ_opt_iter.minimizer)

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
    while norm(abs.(x-x0))>=0.0000001 && i < 100
        println("x=$x")
        x0 = x
        x = conlin_iter(V, dVdx, g, dgdx, x, x_range)
    end
end

# # Целевая функция
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
x0=[2.0,1.0]

conlin(
    V,
    dVdx,
    [g],
    [dgdx],
    x0,
    x_range    
)



# Целевая функция
# V(x)= x[1]+x[2]
# dVdx1(x) = 1
# dVdx2(x) = 1
# dVdx = [dVdx1, dVdx2]

# # Ограничения
# g1(x) = 1 - x[1]^2 * x[2] / 20
# dg1dx1(x) = -x[1] * x[2] / 10
# dg1dx2(x) = -x[1]^2 / 20

# g2(x) = 1 - (x[1] + x[2]-5)*(x[1]+x[2]-5)/30-(x[1]-x[2]-12)*(x[1]-x[2]-12)/120
# dg2dx1(x) = 1 / 15 * (5 - x[1] - x[2]) + 1 / 60 * (12 - x[1] + x[2])
# dg2dx2(x) = 1 / 15 * (5 - x[1] - x[2]) + 1 / 60 * (-12 + x[1] - x[2])

# g3(x) = 1 - 80 / (x[1]^2 + 8 * x[2] + 5)
# dg3dx1(x) = (160 * x[1]) / ((5 + x[1]^2 + 8 * x[2]) * (5 + x[1]^2 + 8 * x[2]))
# dg3dx2(x) =  640 / ((5 + x[1]^2 + 8 * x[2]) * (5 + x[1]^2 + 8 * x[2]))

# g4(x) = -x[1] + x[2]^2 / 15 + 7 / 15
# dg4dx1(x) = -1
# dg4dx2(x) = (2 * x[2]) / 15

# g5(x) = x[1] - x[2] / 5 - 5
# dg5dx1(x) = 1
# dg5dx2(x) = -(1 / 5)

# g_vector = [g1, g2, g3, g4, g5]
# dgdx = [
#     [dg1dx1, dg1dx2], 
#     [dg2dx1, dg2dx2], 
#     [dg3dx1, dg3dx2],
#     [dg4dx1, dg4dx2],
#     [dg5dx1, dg5dx2]
# ]

# x1_range=[0.0,10.0]
# x2_range=[0.0,10.0]
# x_range = [x1_range, x2_range]
# x0=[5.0,5.0]

# conlin(
#     V,
#     dVdx,
#     g_vector,
#     dgdx,
#     x0,
#     x_range    
# )