# Целевая функция
V(x)= 4 / (3*x[1]) + 1/x[2]
dVdx1(x) = -4/(3*x[1]^2)
dVdx2(x) = -1/(x[1]^2)
dVdx = [dVdx1, dVdx2]

# Функция ограничений
g(x) = 4 / sqrt(3)*x[1]+ sqrt(3)*x[2]-1
dgdx1(x) = 4/sqrt(3)
dgdx2(x) = sqrt(3)
dgdx = [dgdx1, dgdx2]

x1_range=[0.0,100.0]
x2_range=[0.0,100.0]
x_range = [x1_range, x2_range]
x0=[0.4,0.4]

conlin(
    V,
    dVdx,
    g,
    dgdx,
    x0,
    x_range    
)
# Решение: [0.247, 0.247]