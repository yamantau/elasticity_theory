import numpy as np
import matplotlib.pyplot as plt

# шаг
h=0.01

# начальные значения
x_0=-1
y_0=-1

# кол-во итераций/кол-во x-ов
n = 100

# задаем наши функции х(т) и у(т)
def f_x(t, x):
    return - np.log(t) * x
def f_y(t, y):
    return np.exp(t) * y

# метод рунге - кутты
# зададим коэфиценты таблицы(серединные и нижние и левые)
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

# метод, который осуществляет рассчет значений по методу рунге
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

# каждой точке по иксу присваивает n значений по рунге
def get_x_for_trajectory_runge(x0):
    return runge_method(x0, h, n, f_x, a, b, c)

# каждой этой точке находим траекторию по рунге
coords_each_x_of_body = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]

# тоже самое для игрека
massivy_t = runge_method(y_0,h,n,f_y,a,b,c)

# создаем траекторию для каждой точки
for each_coord in coords_each_x_of_body:
    plt.grid()
    plt.plot(get_x_for_trajectory_runge(each_coord), massivy_t, color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')

plt.savefig('plots/trajectory.svg', format='svg', dpi=1200)

# строим поле векторов и линии тока
def getPlotOfField(time):
    X = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    Y = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])

    i = 0
    j = 0

    # создаем пустой массив 100 на 100
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