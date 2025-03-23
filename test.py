from mathkat import Funcion
import numpy as np
# ===================== Funciones =====================


f1 = lambda x: x**2 + 2*x + 1
grad_f1 = lambda x: np.array([2*x + 2])

f2 = lambda x1,x2: x1**2 + (2*x2**2)
grad_f2 = lambda x1,x2: np.array([2*x1, 4*x2])

f3 = lambda x1,x2,x3: x1**2 + x2**2 + 2*x3**2
grad_f3 = lambda x1,x2,x3: np.array([2*x1, 2*x2, 4*x3])

# ================= Función 1 =================

x_0 = np.array([10])
alpha = 0.1
v_0 = np.array([0])
iteraciones = 50
epsilon = 0.001
eta = 0.9

f1 = Funcion(f1,grad_f1,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION SIN MOMENTUM")
f1.desenso_gradiente_simple()
print("Momentum F1")
f1.desenso_gradiente_momentum()

# # ================= Función 2 =================

x_0 = np.array([-5,-2])
alpha = 0.1
v_0 = np.array([0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9

f2 = Funcion(f2,grad_f2,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION SIN MOMENTUM")
f2.desenso_gradiente_simple()
print("Momentum F2")
f2.desenso_gradiente_momentum()

# # ================= Función 3 =================
x_0 = np.array([-1,-1,-1])
alpha = 0.1
v_0 = np.array([0,0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9
f3 = Funcion(f3,grad_f3,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION SIN MOMENTUM")
f3.desenso_gradiente_simple()
print("Momentum F3")
f3.desenso_gradiente_momentum()