# Целевая функция
V(x)= x[1] + x[2]
dVdx1(x) = 1
dVdx2(x) = 1
dVdx = [dVdx1, dVdx2]

# Функция ограничений
g(x) = 1 / (x[1]^3) + 7 / (x[2]^3) - 1
dgdx1(x) = -3 /( x[1]^3)
dgdx2(x) = -21 /( x[2]^3)
dgdx = [dgdx1, dgdx2]

x1_range=[0.1,10.0]
x2_range=[0.1,10.0]
x_range = [x1_range, x2_range]
x0=[1,2]

conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    x_range    
)
# Решение: [1.380, 2.244 ]