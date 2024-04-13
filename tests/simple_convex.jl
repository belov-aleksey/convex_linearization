# Целевая функция
V(x)= (x[1]-3)^2 + (x[2]+1)^2
dVdx1(x) = 2*(x[1]-3)
dVdx2(x) = 2*(x[2]+1)
dVdx = [dVdx1, dVdx2]

# Функция ограничений
g(x) = x[1]+x[2]-1.5
dgdx1(x) = 1
dgdx2(x) = 1
dgdx = [dgdx1, dgdx2]

x1_range=[0.0,1.0]
x2_range=[-2.0,1.0]
x_range = [x1_range, x2_range]
x0=[0.5,0.5]

conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    x_range    
)
# Решение: [1.0, -1.0]