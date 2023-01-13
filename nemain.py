from models.material_points import Material_points
from models.material_body import Material_body
from models.point_trajectory import Point_trajectory
import numpy as np
import matplotlib.pyplot as plt

# шаг
h=0.01

# начальные значения
x_0=-1
y_0=-1

# кол-во итераций/кол-во x-ов
n = 100

def f_x(t, x):
    return - np.log(t) * x

def f_y(t, y):
    return np.exp(t) * y

def create_material_body(time):
    material_points = []
    i = -1.1
    while i < -0.11:
        i += 0.1

        Vx = f_x(time, i)
        Vy = f_y(time, -1)

        mp = Material_points(i, -1, Vx, Vy)

        material_points.append(mp)

    return Material_body(material_points, time)

body = create_material_body(0.1)

a = np.array([
    [0, 0, 0, 0],  # a0
    [0, 0, 0, 0],  # a1
    [0, 1/3, 0, 0],  # a2
    [0, 0, 2/3, 0],  # a3
])

# нижняя часть
b = np.array([0, 1 / 4, 0, 3/4])

# левая часть
c = np.array([0, 0, 1/3, 2/3])

def runge_method(x_0, h, n, func, a, b, c):
    x_t = [x_0]
    t = 0.0001
    for i in range(n):
        x_n = x_t[i]
        k1 = func(t, x_n)
        k2 = func(t + c[2] * h, x_n + a[2, 1] * h * k1)
        k3 = func(t + c[3] * h, x_n + a[3, 1] * h * k1 + a[3, 2] * h * k2)
        x_t.append(x_n + h * (k1 * b[1] + k2 * b[2] + k3 * b[3]))
        t += h
    return x_t

def get_trajectory(point):
    return runge_method(point.Ax, h, n, f_x, a, b, c)

i = 0
while i < 10:
    coord_x = get_trajectory(body.material_points[i])
    trajectory = Point_trajectory(coord_x, runge_method(-1, h, n, f_y, a, b, c))
    plt.plot(trajectory.x_coords, trajectory.y_coords, color='r')
    i += 1

plt.xlabel('x')
plt.ylabel('y')
plt.axis('tight')
# plt.show()
plt.savefig('plots/trajectory.svg', format='svg')

def getPlotOfField(time):
    X = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    Y = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])

    i = 0
    j = 0

    # создаем пустой массив 10 на 10
    Ux = np.zeros((10, 10))
    Vy = np.zeros((10, 10))

    # перебираеми каждый элемент, что присвоить две скорости
    while i < 10:
        Ynow = Y[i]
        while j < 10:
            Xnow = X[j]

            # считаем скорость для каждой точки в каждый момент времени по x и y
            Ux[i][j] = f_x(time, Xnow)
            Vy[i][j] = f_y(time, Ynow)
            j += 1
        j = 0
        i += 1

    # строим график
    fig, ax = plt.subplots()

    ax.quiver(X, Y, Ux, Vy)
    ax.streamplot(X, Y, Ux, Vy, color='b')

    fig.set_figwidth(8)  # ширина
    fig.set_figheight(8)  # высота

    # сохраняем
    plt.savefig('plots/plots_in_time_' + str(round(time, 1)) + '_sec.svg', format='svg', dpi=1200)

getPlotOfField(0.1)

# перебираем время с шагом = 0.1 (для графика)
k = 0.1
while k < 1.1:
    getPlotOfField(k)
    k += 0.1



#


