# Clase 2 — Ejemplos guiados y ejercicios

**Módulo:** Deep Learning  
**Libro base:** Simon J. D. Prince, *Understanding Deep Learning* (2023), capítulos 3 y 4.  
**Notebooks:** `3_1_Shallow_Networks_I.ipynb`, `3_2_Shallow_Networks_II.ipynb` y `4_3_Deep_Networks.ipynb`.  
**Duración del bloque:** 45 minutos.  
**Prerrequisitos:** operaciones matriciales, funciones afines y ReLU.

> **Uso docente:** mostrar los enunciados y pedir predicciones antes de revelar las soluciones. Los ejemplos 1–3 se resuelven en pizarra; los ejercicios 1–5 se pueden asignar a parejas. Las respuestas aparecen al final.

## Objetivos del bloque

1. Distinguir preactivación, activación y salida de una red.
2. Calcular manualmente una red shallow 1–3–1.
3. Convertir una capa a notación matricial y verificar dimensiones.
4. Relacionar cambios de parámetros con cambios de la función representada.
5. Preparar las tareas `TODO` de los notebooks oficiales.

---

## Ejemplo guiado 1 — Una neurona con ReLU (10 minutos)

### Enunciado

Una neurona recibe dos entradas: $x_1=2$, $x_2=3$. Sus pesos son $w_1=0.5$, $w_2=-1$ y el sesgo $b=1$.

1. Calcular la preactivación $z$.
2. Aplicar la función $\operatorname{ReLU}(z)=\max(0,z)$.
3. Repetir con $b=3$.
4. Explicar qué cambió y por qué.

### Resolución guiada

$$z=w_1x_1+w_2x_2+b=(0.5)(2)+(-1)(3)+1=-1.$$

$$a=\operatorname{ReLU}(-1)=0.$$

Si $b=3$, entonces $z=1-3+3=1$ y $a=1$.

**Interpretación:** el sesgo desplaza la preactivación. ReLU produce cero cuando la preactivación es negativa y conserva los valores positivos.

### Verificación con NumPy

```python
import numpy as np

def ReLU(preactivation):
    return np.maximum(0, preactivation)

x = np.array([2.0, 3.0])
w = np.array([0.5, -1.0])
for b in [1.0, 3.0]:
    z = w @ x + b
    print(f"b={b:.1f}, z={z:.1f}, ReLU(z)={ReLU(z):.1f}")
```

**Conexión con el notebook 3.1:** la celda de ReLU tiene un `TODO` que reemplaza `np.zeros_like(preactivation)` por una operación elemento a elemento, por ejemplo `np.maximum(0, preactivation)` o `preactivation.clip(min=0)`.

---

## Ejemplo guiado 2 — Red shallow 1–3–1 (10 minutos)

### Enunciado

Considerar la arquitectura con una entrada $x$, tres neuronas ocultas y una salida:

$$z_i=\theta_{i0}+\theta_{i1}x,\quad a_i=\operatorname{ReLU}(z_i),\quad y=\phi_0+\phi_1a_1+\phi_2a_2+\phi_3a_3.$$

Usar **los parámetros originales de la celda de experimentación del notebook 3.1**:

| Neurona | $\theta_{i0}$ | $\theta_{i1}$ | $\phi_i$ |
|---|---:|---:|---:|
| 1 | 0.3 | −1.0 | 2.0 |
| 2 | −1.0 | 2.0 | −1.0 |
| 3 | −0.5 | 0.65 | 7.0 |

Sesgo de salida: $\phi_0=-0.3$.

Calcular todas las preactivaciones, activaciones, activaciones ponderadas y la salida para $x=0.5$.

### Resolución guiada

**Preactivaciones:**

$$z_1=0.3-0.5=-0.2;\qquad z_2=-1+2(0.5)=0;\qquad z_3=-0.5+0.65(0.5)=-0.175.$$

**Activaciones:** $a_1=a_2=a_3=0$.

**Contribuciones ponderadas:** $\phi_1a_1=0$, $\phi_2a_2=0$, $\phi_3a_3=0$.

**Salida:**

$$\boxed{y=-0.3}.$$

**Discusión:** ¿significa esto que la red siempre devuelve $-0.3$? No. Solo ocurre en entradas para las cuales las tres activaciones son cero. Pida calcular después $x=0$ y $x=1$ para comparar.

### Verificación con NumPy

```python
import numpy as np
x = np.array([0.0, 0.5, 1.0])
theta_0 = np.array([0.3, -1.0, -0.5])
theta_1 = np.array([-1.0, 2.0, 0.65])
phi = np.array([2.0, -1.0, 7.0])
phi_0 = -0.3
z = theta_0[:, None] + theta_1[:, None] * x[None, :]
a = np.maximum(0, z)
y = phi_0 + phi @ a
print("preactivaciones:\n", z)
print("activaciones:\n", a)
print("salidas:", y)
```

**Conexión con el notebook 3.1:** la función `shallow_1_1_3` contiene `TODO` para calcular preactivaciones, ponderar activaciones y construir la salida final. Las figuras intermedias de `plot_neural(..., plot_all=True)` corresponden a esas tres etapas.

---

## Ejemplo guiado 3 — Una capa en forma matricial (10 minutos)

### Enunciado

Una capa tiene dos entradas y tres neuronas:

$$x=\begin{bmatrix}2\\1\end{bmatrix},\quad
W=\begin{bmatrix}1&-1\\2&0\\0.5&1\end{bmatrix},\quad
b=\begin{bmatrix}0\\1\\-1\end{bmatrix}.$$

1. Identificar las dimensiones de $x$, $W$ y $b$.
2. Calcular $z=Wx+b$.
3. Calcular $h=\operatorname{ReLU}(z)$.
4. Explicar por qué $W$ tiene tres filas y dos columnas.

### Resolución guiada

Dimensiones: $x\in\mathbb{R}^{2\times1}$, $W\in\mathbb{R}^{3\times2}$, $b\in\mathbb{R}^{3\times1}$.

$$Wx=\begin{bmatrix}1\\4\\2\end{bmatrix},\qquad
z=Wx+b=\begin{bmatrix}1\\5\\1\end{bmatrix}.$$

Como todas las preactivaciones son positivas:

$$\boxed{h=\begin{bmatrix}1\\5\\1\end{bmatrix}}.$$

**Interpretación:** cada fila de $W$ representa los pesos de una neurona de destino; cada columna corresponde a una entrada.

### Verificación con NumPy

```python
import numpy as np
x = np.array([[2.0], [1.0]])
W = np.array([[1.0, -1.0], [2.0, 0.0], [0.5, 1.0]])
b = np.array([[0.0], [1.0], [-1.0]])
z = W @ x + b
h = np.maximum(0, z)
print(x.shape, W.shape, b.shape, z.shape)
print("z:\n", z)
print("h:\n", h)
```

**Conexión con el notebook 4.3:** sus `TODO` requieren construir los vectores `beta` y las matrices `Omega` con las dimensiones correctas. La misma operación se aplica a muchas entradas si se colocan como columnas de una matriz.

---

## Ejercicios para los estudiantes (10 minutos)

### Ejercicio 1 — Activaciones elementales (2 min)

Aplicar ReLU a $z=[-4,-0.5,0,1.5,3]$. Escribir el resultado y explicar qué ocurre con los valores negativos.

### Ejercicio 2 — Cambios de parámetros (2 min)

En la red del ejemplo guiado 2, ¿qué sucede con la salida si se incrementa $\phi_0$ en $0.5$? ¿Y qué sucede si se multiplican simultáneamente $\phi_1,\phi_2,\phi_3$ por $0.5$? Escribir la nueva expresión de salida, sin necesidad de graficar.

### Ejercicio 3 — Dos entradas y una neurona (2 min)

Una neurona de una red con dos entradas calcula $z=-4+0.9x_1+0x_2$. Evaluar $z$ y ReLU para $(x_1,x_2)=(2,7)$ y $(6,7)$. Explicar por qué cambiar $x_2$ no afecta esta neurona.

**Conexión:** primera neurona de la red `shallow_2_1_3` del notebook 3.2, con sus parámetros originales.

### Ejercicio 4 — Conteo de parámetros (2 min)

Calcular los parámetros entrenables de estas arquitecturas completamente conectadas, suponiendo un sesgo por neurona de destino:

- A: $2\rightarrow 8\rightarrow1$.
- B: $2\rightarrow6\rightarrow4\rightarrow1$.

¿Cuál tiene mayor profundidad? ¿Cuál tiene más parámetros?

### Ejercicio 5 — Dimensiones en una red profunda (2 min)

El notebook 4.3 propone una red con $D_i=4$, $D_1=5$, $D_2=2$, $D_3=4$, $D_o=1$ y cuatro observaciones en columnas. Indicar las dimensiones de `x`, `Omega_0`, `beta_0`, `Omega_1`, `beta_1`, `Omega_2`, `beta_2`, `Omega_3`, `beta_3` y `y`.

---

## Solucionario breve (5 minutos de puesta en común)

**Ejercicio 1:** $[0,0,0,1.5,3]$. ReLU anula los valores negativos y conserva los no negativos.

**Ejercicio 2:** al sumar $0.5$ a $\phi_0$, la función completa se desplaza $0.5$ unidades hacia arriba. Al multiplicar los tres pesos de salida por $0.5$, $y_{nuevo}=\phi_0+0.5\sum_i\phi_i a_i=\phi_0+0.5(y_{original}-\phi_0)$. La variación se calcula respecto al sesgo de salida, no respecto al origen.

**Ejercicio 3:** para $(2,7)$, $z=-2.2$ y ReLU $=0$; para $(6,7)$, $z=1.4$ y ReLU $=1.4$. El coeficiente de $x_2$ es cero, por lo que esta neurona no depende de esa variable.

**Ejercicio 4:** A: $8(2+1)+1(8+1)=33$; B: $6(2+1)+4(6+1)+1(4+1)=51$. B tiene más capas ocultas y más parámetros. Esto **no garantiza** un mejor desempeño.

**Ejercicio 5:**

| Variable | Dimensión |
|---|---|
| `x` | (4, 4) |
| `Omega_0` | (5, 4) |
| `beta_0` | (5, 1) |
| `Omega_1` | (2, 5) |
| `beta_1` | (2, 1) |
| `Omega_2` | (4, 2) |
| `beta_2` | (4, 1) |
| `Omega_3` | (1, 4) |
| `beta_3` | (1, 1) |
| `y` | (1, 4) |

**Nota:** los vectores de sesgos se difunden (*broadcasting*) a las cuatro columnas de observaciones durante las sumas matriciales.

---

## Ejercicios de ampliación para el laboratorio

Si el grupo termina antes, utilizar estas consignas **dentro de los notebooks originales**, sin alterar sus parámetros de referencia hasta haber registrado la salida inicial.

1. **Notebook 3.1, celda de experimentos:** ejecutar las ocho consignas del `TODO`, formulando por escrito una predicción antes de cada cambio. Atención especial a las preguntas sobre los *joints* y sobre escalar conjuntamente parámetros de una neurona.
2. **Notebook 3.1, pérdida:** implementar `np.sum((y_train - y_predict)**2)`. Verificar que el notebook muestra aproximadamente `9.385` para los parámetros originales y buscar una mejora modificando primero `phi_0`. El propio notebook menciona un valor de referencia de `0.181` obtenido manualmente; no se exige alcanzarlo.
3. **Notebook 3.2:** completar `shallow_2_1_3`, visualizar mapas de calor y localizar regiones con distinta combinación de neuronas activas. Después implementar `shallow_2_2_3` con dos salidas que comparten las tres activaciones ocultas.
4. **Notebook 4.3:** completar las matrices `beta_0`, `Omega_0`, `beta_1` y `Omega_1` de la primera red. Después completar la composición de dos redes y verificar que las gráficas coinciden con la implementación escalar.
5. **Notebook 4.3, red de tres capas ocultas:** completar las dimensiones de las ocho matrices/vectores aleatorios y verificar que la salida tiene forma `(1, 4)`.

## Lista de cotejo para el docente

- [ ] El estudiante diferencia $z$ (preactivación) de $a$ (activación).
- [ ] Aplica ReLU correctamente a escalares y arreglos.
- [ ] Calcula una salida 1–3–1 sin confundir $\theta$ con $\phi$.
- [ ] Justifica las dimensiones de una matriz de pesos.
- [ ] Cuenta parámetros por capa y suma los totales.
- [ ] Puede predecir qué cambia cuando modifica un sesgo o un peso.
- [ ] Comprueba sus resultados manuales con NumPy.

**Cierre sugerido:** «Ya sabemos cómo una arquitectura produce salidas. En la siguiente clase estudiaremos cómo medir matemáticamente el error de esas salidas mediante funciones de pérdida».
