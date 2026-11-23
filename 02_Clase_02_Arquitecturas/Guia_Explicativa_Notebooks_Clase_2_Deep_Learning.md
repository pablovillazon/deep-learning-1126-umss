# Guía explicativa de laboratorio — Clase 2: Arquitectura de redes neuronales

**Curso:** Deep Learning · **Texto base:** Simon J. D. Prince, *Understanding Deep Learning*, capítulos 3 y 4  
**Fuente de trabajo:** los tres archivos `.ipynb` adjuntados por el docente (no versiones supuestas de Internet).  
**Notebooks:** `3_1_Shallow_Networks_I.ipynb`, `3_2_Shallow_Networks_II.ipynb`, `4_3_Deep_Networks.ipynb`.

## Cómo usar esta guía

Esta guía sigue **el orden real de las celdas** de los archivos originales. Para cada bloque se explica su propósito, las operaciones de Python/NumPy, la matemática correspondiente, el resultado que debe observarse y los ejercicios `TODO`. Los bloques identificados como **«Solución propuesta»** son material didáctico añadido: no deben confundirse con código que ya estuviera resuelto en el notebook. Las celdas auxiliares de dibujo se explican por grupos, pero se conserva su identificación.

**Convenciones:** `pre_i` denota la preactivación de la neurona oculta `i`; `act_i` su activación; `w_act_i` su contribución ponderada a la salida; `theta` contiene pesos y sesgos de entrada; `phi` contiene los parámetros de salida. En el notebook 4.3, `Omega` es la matriz de pesos y `beta` el vector de sesgos. Se usa ReLU: $\operatorname{ReLU}(z)=\max(0,z)$.

> **Recomendación de trabajo:** antes de ejecutar una celda con `TODO`, formular una predicción, completar el código, ejecutar y contrastar con el resultado observado. En las actividades de modificación de parámetros, conservar una copia de los valores originales.

---

# Parte I — Notebook 3.1: *Shallow neural networks I*

**Objetivo original:** familiarizarse con redes poco profundas con **una entrada**, **tres neuronas ocultas** y **una salida**; reproducir el ejemplo asociado a la figura 3.3 del libro, experimentar con los parámetros y calcular una pérdida por mínimos cuadrados.

## Celdas 0–3 — Acceso, propósito, bibliotecas y arquitectura

- **Celda 0 (Markdown):** contiene un enlace para abrir el notebook en Google Colab; no interviene en los cálculos.
- **Celda 1 (Markdown):** presenta el propósito y solicita ejecutar las celdas en orden, completar los `TODO` y responder las preguntas.
- **Celda 2 (código):** importa `numpy as np` para arreglos y operaciones numéricas y `matplotlib.pyplot as plt` para las figuras.
- **Celda 3 (Markdown):** introduce la arquitectura $1\to3\to1$ descrita en la sección 3.1.

**Código original — celda 2:**

```python
# Imports math library
import numpy as np
# Imports plotting library
import matplotlib.pyplot as plt
```

**Lectura matemática:** para una entrada escalar $x$, cada neurona oculta calcula $z_i=\theta_{i0}+\theta_{i1}x$ y $a_i=\operatorname{ReLU}(z_i)$. La salida es $y=\phi_0+\phi_1a_1+\phi_2a_2+\phi_3a_3$. En total hay **10 parámetros**: seis `theta` y cuatro `phi`.

## Celda 4 — Implementar ReLU (`TODO`)

**Código original — celda 4:**

```python
# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  # TODO write code to implement the ReLU and compute the activation at the
  # hidden unit from the preactivation
  # This should work on every element of the ndarray "preactivation" at once
  # One way to do this is with the ndarray "clip" function
  # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html
  activation = np.zeros_like(preactivation);

  return activation
```

### Qué hace la implementación inicial

`np.zeros_like(preactivation)` crea un arreglo de ceros con la misma forma y tipo que la entrada. **No implementa ReLU**: anula también los valores positivos. Es una implementación provisional intencional para que el estudiante la sustituya.

### Qué solicita el `TODO`

Calcular $a_i=\max(0,z_i)$ **elemento a elemento**, sin recorrer el arreglo con `for`. El comentario del notebook propone `ndarray.clip`.

**Solución propuesta — siguiendo la pista del notebook:**

```python
def ReLU(preactivation):
    activation = preactivation.clip(0.0)
    return activation
```

**Alternativa equivalente:** `activation = np.maximum(0, preactivation)`. `clip(0.0)` impone un límite inferior igual a cero, sin imponer límite superior. En ambos casos se conservan la forma y la vectorización del arreglo.

**Prueba mínima:**

```python
z = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
print(ReLU(z))  # [0. 0. 0. 2. 5.]
```

**Error conceptual que conviene discutir:** si todas las activaciones ocultas se anulan, la red solo devuelve el sesgo de salida $y=\phi_0$, sin dependencia de $x$.

## Celda 5 — Graficar ReLU

**Código original — celda 5:**

```python
# Make an array of inputs
z = np.arange(-5,5,0.1)
RelU_z = ReLU(z)

# Plot the ReLU function
fig, ax = plt.subplots()
ax.plot(z,RelU_z,'r-')
ax.set_xlim([-5,5]);ax.set_ylim([-5,5])
ax.set_xlabel('z'); ax.set_ylabel('ReLU[z]')
ax.set_aspect('equal')
plt.show()
```

- `np.arange(-5,5,0.1)` crea muestras desde −5 hasta antes de 5, espaciadas 0.1 unidades.
- `RelU_z = ReLU(z)` evalúa **todas las muestras a la vez**.
- `plt.subplots()` crea una figura y un eje; `ax.plot(z, RelU_z, 'r-')` dibuja la curva roja continua.
- `set_xlim` y `set_ylim` fijan el encuadre; `set_aspect('equal')` iguala la escala visual de ambos ejes.
- **Resultado esperado:** la curva es horizontal en cero para $z<0$ y coincide con la recta $a=z$ para $z>0$. Si aparece una recta horizontal en cero a ambos lados, la celda 4 sigue incompleta.

## Celda 6 — Construir `shallow_1_1_3` (tres `TODO`)

**Código original — celda 6:**

```python
# Define a shallow neural network with, one input, one output, and three hidden units
def shallow_1_1_3(x, activation_fn, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):
  # TODO Replace the code below to compute the three initial lines
  # from the theta parameters (i.e. implement equations at bottom of figure 3.3a-c).  These are the preactivations
  pre_1 = np.zeros_like(x)
  pre_2 = np.zeros_like(x)
  pre_3 = np.zeros_like(x)

  # Pass these through the ReLU function to compute the activations as in
  # figure 3.3 d-f
  act_1 = activation_fn(pre_1)
  act_2 = activation_fn(pre_2)
  act_3 = activation_fn(pre_3)

  # TODO Replace the code below to weight the activations using phi1, phi2 and phi3
  # To create the equivalent of figure 3.3 g-i
  w_act_1 = np.zeros_like(x)
  w_act_2 = np.zeros_like(x)
  w_act_3 = np.zeros_like(x)

  # TODO Replace the code below to combining the weighted activations and add
  # phi_0 to create the output as in figure 3.3 j
  y = np.zeros_like(x)

  # Return everything we have calculated
  return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3
```

### Etapa A — Preactivaciones

Cada neurona oculta produce una recta: $z_1=\theta_{10}+\theta_{11}x$, $z_2=\theta_{20}+\theta_{21}x$, $z_3=\theta_{30}+\theta_{31}x$. **Solución propuesta:**

```python
pre_1 = theta_10 + theta_11 * x
pre_2 = theta_20 + theta_21 * x
pre_3 = theta_30 + theta_31 * x
```

### Etapa B — Activaciones (ya implementada en el original)

`act_i = activation_fn(pre_i)` aplica la función recibida como argumento. Esta decisión permite reutilizar la misma red con otras activaciones, sin modificar su estructura. Con ReLU, cada recta queda recortada a cero donde es negativa.

### Etapa C — Contribuciones ponderadas

El segundo `TODO` pide multiplicar la activación de cada unidad por el peso de salida correspondiente:

```python
w_act_1 = phi_1 * act_1
w_act_2 = phi_2 * act_2
w_act_3 = phi_3 * act_3
```

### Etapa D — Salida de la red

El tercer `TODO` pide sumar las tres contribuciones y el sesgo de salida:

```python
y = phi_0 + w_act_1 + w_act_2 + w_act_3
```

### Implementación completa propuesta

```python
def shallow_1_1_3(x, activation_fn, phi_0, phi_1, phi_2, phi_3,
                  theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):
    pre_1 = theta_10 + theta_11 * x
    pre_2 = theta_20 + theta_21 * x
    pre_3 = theta_30 + theta_31 * x
    act_1 = activation_fn(pre_1)
    act_2 = activation_fn(pre_2)
    act_3 = activation_fn(pre_3)
    w_act_1 = phi_1 * act_1
    w_act_2 = phi_2 * act_2
    w_act_3 = phi_3 * act_3
    y = phi_0 + w_act_1 + w_act_2 + w_act_3
    return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3
```

**Por qué se devuelven diez resultados:** el notebook necesita tanto la predicción `y` como los nueve arreglos intermedios para construir los paneles de la figura 3.3. Esto hace observable el funcionamiento interno de la red.

## Celda 7 — `plot_neural`: cómo interpretar los 9 paneles

**Código original — celda 7:**

```python
# Plot the shallow neural network.  We'll assume input in is range [0,1] and output [-1,1]
# If the plot_all flag is set to true, then we'll plot all the intermediate stages as in Figure 3.3
def plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=False, x_data=None, y_data=None):

  # Plot intermediate plots if flag set
  if plot_all:
    fig, ax = plt.subplots(3,3)
    fig.set_size_inches(8.5, 8.5)
    fig.tight_layout(pad=3.0)
    ax[0,0].plot(x,pre_1,'r-'); ax[0,0].set_ylabel('Preactivation')
    ax[0,1].plot(x,pre_2,'b-'); ax[0,1].set_ylabel('Preactivation')
    ax[0,2].plot(x,pre_3,'g-'); ax[0,2].set_ylabel('Preactivation')
    ax[1,0].plot(x,act_1,'r-'); ax[1,0].set_ylabel('Activation')
    ax[1,1].plot(x,act_2,'b-'); ax[1,1].set_ylabel('Activation')
    ax[1,2].plot(x,act_3,'g-'); ax[1,2].set_ylabel('Activation')
    ax[2,0].plot(x,w_act_1,'r-'); ax[2,0].set_ylabel('Weighted Act')
    ax[2,1].plot(x,w_act_2,'b-'); ax[2,1].set_ylabel('Weighted Act')
    ax[2,2].plot(x,w_act_3,'g-'); ax[2,2].set_ylabel('Weighted Act')

    for plot_y in range(3):
      for plot_x in range(3):
        ax[plot_y,plot_x].set_xlim([0,1]);ax[plot_x,plot_y].set_ylim([-1,1])
        ax[plot_y,plot_x].set_aspect(0.5)
      ax[2,plot_y].set_xlabel('Input, $x$');
    plt.show()

  fig, ax = plt.subplots()
  ax.plot(x,y)
  ax.set_xlabel('Input, $x$'); ax.set_ylabel('Output, $y$')
  ax.set_xlim([0,1]);ax.set_ylim([-1,1])
  ax.set_aspect(0.5)
  if x_data is not None:
    ax.plot(x_data, y_data, 'mo')
    for i in range(len(x_data)):
      ax.plot(x_data[i], y_data[i],)
  plt.show()
```

- `plot_all=False` evita dibujar los nueve gráficos intermedios; cuando es `True`, se crea una cuadrícula `3×3`.
- **Fila 1:** `pre_1`, `pre_2`, `pre_3` (rectas originales de cada neurona).
- **Fila 2:** `act_1`, `act_2`, `act_3` (rectas después de ReLU).
- **Fila 3:** `w_act_1`, `w_act_2`, `w_act_3` (activaciones multiplicadas por los pesos `phi`).
- Se dibuja después una figura independiente con la salida final `y`.
- `x_data` y `y_data` son opcionales: si se suministran, los puntos de entrenamiento aparecen superpuestos en color magenta.
- `set_xlim`, `set_ylim` y `set_aspect` controlan los ejes, no los valores de la red.

**Detalle del código original:** en el doble bucle aparece `ax[plot_x,plot_y].set_ylim(...)`, con los índices invertidos respecto de `ax[plot_y,plot_x]`. Como el bucle recorre las nueve posiciones, finalmente ajusta todos los paneles, pero por claridad puede corregirse a `ax[plot_y,plot_x].set_ylim([-1,1])`. La línea `ax.plot(x_data[i], y_data[i],)` dentro del bucle de puntos tampoco es necesaria para obtener los marcadores, ya dibujados mediante `'mo'`.

## Celdas 8–10 — Ejecutar la red y contrastar con la figura de referencia

**Código original — celda 8:**

```python
# Now lets define some parameters and run the neural network
theta_10 =  0.3 ; theta_11 = -1.0
theta_20 = -1.0  ; theta_21 = 2.0
theta_30 = -0.5  ; theta_31 = 0.65
phi_0 = -0.3; phi_1 = 2.0; phi_2 = -1.0; phi_3 = 7.0

# Define a range of input values
x = np.arange(0,1,0.01)

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)
# And then plot it
plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True)
```

**Valores concretos del notebook:**

| Neurona | Sesgo `theta_i0` | Peso `theta_i1` | Peso de salida `phi_i` | Preactivación |
|---|---:|---:|---:|---|
| 1 | 0.3 | −1.0 | 2.0 | $z_1=0.3-x$ |
| 2 | −1.0 | 2.0 | −1.0 | $z_2=-1+2x$ |
| 3 | −0.5 | 0.65 | 7.0 | $z_3=-0.5+0.65x$ |

Además, $\phi_0=-0.3$. La salida es:

$$y(x)=-0.3+2\operatorname{ReLU}(0.3-x)-\operatorname{ReLU}(-1+2x)+7\operatorname{ReLU}(-0.5+0.65x).$$

Los puntos de cambio aparecen cuando las preactivaciones cruzan cero: $x=0.3$, $x=0.5$ y $x=0.5/0.65\approx0.76923$. La celda 8 calcula la red para `x=np.arange(0,1,0.01)` y llama al trazador con `plot_all=True`.

- **Celda 9:** contiene una **imagen SVG incrustada** como referencia visual del resultado esperado; no es código que haya que modificar.
- **Celda 10:** repite los valores iniciales para preparar los experimentos de la siguiente celda.

**Comprobación útil:** el segundo tramo (entre 0.3 y 0.5) tiene pendiente cero: en esa región las tres activaciones son cero y $y=-0.3$.

## Celda 11 — Ocho preguntas `TODO` sobre parámetros

**Código original — celda 11:**

```python
# TODO
# 1. Predict what effect changing phi_0 will have on the network.

# 2. Predict what effect multiplying phi_1, phi_2, phi_3 by 0.5 would have.  Check if you are correct

# 3. Predict what effect multiplying phi_1 by -1 will have.  Check if you are correct.

# 4. Predict what effect setting theta_20 to -1.2 will have.  Check if you are correct.

# 5. Change the parameters so that there are only two "joints" (including outside the range of the plot)
# There are actually three ways to do this. See if you can figure them all out

# 6. With the original parameters, the second line segment is flat (i.e. has slope zero)
# How could you change theta_10 so that all of the segments have non-zero slopes

# 7. What do you predict would happen if you multiply theta_20 and theta21 by 0.5, and phi_2 by 2.0?
# Check if you are correct.

# 8. What do you predict would happen if you multiply theta_20 and theta21 by -0.5, and phi_2 by -2.0?
# Check if you are correct.

theta_10 =  0.3 ; theta_11 = -1.0
theta_20 = -1.0  ; theta_21 = 2.0
theta_30 = -0.5  ; theta_31 = 0.65
phi_0 = -0.3; phi_1 = 2.0; phi_2 = -1.0; phi_3 = 7.0

# Define a range of input values
x = np.arange(0,1,0.01)

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)
# And then plot it
plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True)
```

### 1. ¿Qué ocurre si cambiamos `phi_0`?

La curva completa se traslada verticalmente por la misma cantidad: $y_{nuevo}(x)=y_{original}(x)+\Delta\phi_0$. No se alteran los puntos de quiebre ni las pendientes.

### 2. ¿Qué ocurre si multiplicamos `phi_1`, `phi_2` y `phi_3` por 0.5?

Se reducen a la mitad las contribuciones de las tres neuronas, **pero no el sesgo**: $y_{nuevo}=\phi_0+0.5(y_{original}-\phi_0)$. La curva se comprime verticalmente respecto de la recta horizontal $y=\phi_0$.

### 3. ¿Qué ocurre si multiplicamos `phi_1` por −1?

La contribución de la neurona 1 cambia de signo. En las regiones donde esa neurona está activa, la diferencia es $\Delta y=-2\phi_1a_1$ (con el `phi_1` original). Fuera de su región activa, no cambia nada.

### 4. ¿Qué ocurre si cambiamos `theta_20` de −1.0 a −1.2?

La preactivación de la segunda neurona pasa de $-1+2x$ a $-1.2+2x$. Su punto de quiebre se mueve de $x=0.5$ a $x=0.6$. Su pendiente preactivación sigue siendo 2. La curva final cambia en la región donde la segunda neurona empieza a contribuir.

### 5. ¿Cómo conseguir solamente dos puntos de quiebre?

Hay varias construcciones posibles, por ejemplo: **(a)** anular el peso de salida de una neurona (`phi_i=0`); **(b)** hacer que una neurona no cruce cero (quede siempre activa o siempre inactiva) en el dominio relevante; **(c)** hacer coincidir dos puntos de quiebre ajustando los parámetros de entrada. **Matiz:** el enunciado pide contar también los puntos fuera del área graficada; si se busca eliminar un quiebre en toda la recta real, hacer que una unidad nunca cruce cero requiere pendiente de entrada cero y sesgo de signo fijo. Desplazar el quiebre fuera de `[0,1]` solo reduce los quiebres visibles, no los globales.

### 6. ¿Cómo evitar el tramo plano cambiando `theta_10`?

Con los parámetros originales, la primera neurona deja de activarse en $x=0.3$ y la segunda comienza en $x=0.5$; entre ambos valores ninguna unidad contribuye, por lo que la pendiente es cero. Si se aumenta `theta_10` por encima de `0.5` (manteniendo `theta_11=-1`), la primera neurona permanece activa hasta después de $x=0.5$, evitando ese intervalo sin unidades activas. Por ejemplo, probar `theta_10=0.6` y verificar las pendientes de todos los tramos. El cruce con la tercera unidad también debe revisarse.

### 7. ¿Qué pasa al multiplicar `theta_20` y `theta_21` por 0.5 y `phi_2` por 2?

Como ReLU es homogénea positiva, $\operatorname{ReLU}(0.5z)=0.5\operatorname{ReLU}(z)$. Por ello $2\phi_2\operatorname{ReLU}(0.5z_2)=\phi_2\operatorname{ReLU}(z_2)$. **La función final no cambia** (salvo posibles redondeos numéricos).

### 8. ¿Qué pasa al multiplicar `theta_20` y `theta_21` por −0.5 y `phi_2` por −2?

Ahora el argumento de ReLU cambia de signo: $\operatorname{ReLU}(-0.5z)$ **no** es $-0.5\operatorname{ReLU}(z)$. Se invierte la región activa de la segunda unidad y la función resultante **sí cambia**. Este contraste explica por qué la homogeneidad positiva no se extiende a factores negativos.

**Actividad sugerida:** tras cada cambio, ejecutar la celda, comparar los nueve paneles y anotar qué elementos permanecieron invariantes (puntos de quiebre, pendientes, desplazamiento o forma).

## Celda 12 — Función de pérdida por mínimos cuadrados

El notebook introduce la pérdida:

$$L(\boldsymbol\phi)=\sum_{i=1}^{I}\left[y_i-f(x_i,\boldsymbol\phi)\right]^2.$$

Cada término mide el cuadrado del error entre la salida observada $y_i$ y la predicción $f(x_i,\boldsymbol\phi)$. La suma agrega los errores de todos los ejemplos. **No es** el error cuadrático medio (MSE), porque el notebook **no divide** entre $I$.

## Celda 13 — Implementar `least_squares_loss` (`TODO`)

**Código original — celda 13:**

```python
# Least squares function
def least_squares_loss(y_train, y_predict):
  # TODO Replace the line below to compute the sum of squared
  # differences between the real values of y and the predicted values from the model f[x_i,phi]
  # (see figure 2.2 of the book)
  # you will need to use the function np.sum
  loss = 0

  return loss
```

**Problema del código provisional:** `loss=0` siempre informa pérdida cero aunque las predicciones sean incorrectas.

**Solución propuesta:**

```python
def least_squares_loss(y_train, y_predict):
    loss = np.sum((y_train - y_predict) ** 2)
    return loss
```

**Verificación independiente:**

```python
real = np.array([1.0, 2.0, 3.0])
pred = np.array([1.0, 1.0, 5.0])
print(least_squares_loss(real, pred))  # 0² + 1² + (−2)² = 5.0
```

## Celda 14 — Datos de entrenamiento, pérdida y ajuste manual

**Código original — celda 14:**

```python
# Now lets define some parameters, run the neural network, and compute the loss
theta_10 =  0.3 ; theta_11 = -1.0
theta_20 = -1.0  ; theta_21 = 2.0
theta_30 = -0.5  ; theta_31 = 0.65
phi_0 = -0.3; phi_1 = 2.0; phi_2 = -1.0; phi_3 = 7.0

# Define a range of input values
x = np.arange(0,1,0.01)

x_train = np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train = np.array([-0.15934537,0.18195445,0.451270150,0.13921448,0.09366691,0.30567674,\
                    0.372291170,0.40716968,-0.08131792,0.41187806,0.36943738,0.3994327,\
                    0.019062570,0.35820410,0.452564960,-0.0183121,0.02957665,-0.24354444, \
                    0.148038840,0.26824970])

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)
# And then plot it
plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True, x_data = x_train, y_data = y_train)

# Run the neural network on the training data
y_predict, *_ = shallow_1_1_3(x_train, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)

# Compute the least squares loss and print it out
loss = least_squares_loss(y_train,y_predict)
print('Your Loss = %3.3f, True value = 9.385'%(loss))

# TODO.  Manipulate the parameters (by hand!) to make the function
# fit the data better and try to reduce the loss to as small a number
# as possible.  The best that I could do was 0.181
# Tip... start by manipulating phi_0.
# It's not that easy, so don't spend too much time on this!
```

- `x_train` y `y_train` contienen **20 pares de entrenamiento** suministrados por el autor.
- Primero se evalúa la red en la malla `x` para visualizar su curva y superponer los puntos de entrenamiento.
- Después se evalúa **en los valores exactos** de `x_train` para obtener `y_predict`; no debe confundirse esta evaluación con la malla de dibujo.
- `y_predict, *_ = shallow_1_1_3(...)` utiliza **desempaquetado extendido**: conserva la primera salida (predicción) e ignora los nueve arreglos intermedios.
- La instrucción `print` compara la pérdida calculada con la referencia **9.385** indicada en el notebook. Esa cifra corresponde a los parámetros iniciales, una vez resueltos correctamente los `TODO`.
- El último `TODO` solicita modificar los parámetros **manualmente** para reducir la pérdida; el autor menciona haber alcanzado aproximadamente **0.181**, sin afirmar que sea un óptimo global.

**Estrategia didáctica:** empezar por `phi_0`, que traslada toda la curva; observar si los puntos quedan sistemáticamente por encima o por debajo. Después ajustar amplitudes (`phi_i`) y, finalmente, posiciones de quiebre (`theta`). Registrar cada intento en una tabla con los parámetros y el valor de pérdida. Esto **no es entrenamiento automático**: es una exploración manual del espacio de parámetros.

**Preguntas de cierre del notebook 3.1:** ¿qué se pierde al dejar ReLU sin implementar?, ¿qué distingue preactivación y activación?, ¿por qué un `phi_i` negativo puede generar una contribución descendente?, ¿qué mide exactamente la pérdida?, ¿por qué un valor de pérdida menor implica mejor ajuste sobre estos datos pero no garantiza generalización?

---

# Parte II — Notebook 3.2: *Shallow neural networks II*

**Objetivo original:** extender la arquitectura shallow a **dos entradas**, visualizar funciones de dos variables y luego producir **dos salidas**. El ejemplo se relaciona con la figura 3.8 del libro.

## Celdas 0–2 — Presentación y bibliotecas

La celda 0 enlaza a Colab; la celda 1 describe el objetivo; la celda 2 importa `numpy` y `matplotlib` de la misma manera que en el notebook 3.1. El cambio conceptual es que la función depende ahora de $x_1$ y $x_2$: su salida escalar se visualiza sobre un plano de entrada.

## Celda 3 — `draw_2D_function` y `plot_neural_2_inputs`

**Código original — celda 3:**

```python
# Code to draw 2D function -- read it so you know what is going on, but you don't have to change it
def draw_2D_function(ax, x1_mesh, x2_mesh, y):
    pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-10,vmax=10.0)
    ax.set_xlabel('x1');ax.set_ylabel('x2')
    levels = np.arange(-10,10,1.0)
    ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')

# Plot the shallow neural network.  We'll assume input in is range [0,10],[0,10] and output [-10,10]
def plot_neural_2_inputs(x1,x2, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3):

  fig, ax = plt.subplots(3,3)
  fig.set_size_inches(8.5, 8.5)
  fig.tight_layout(pad=3.0)
  draw_2D_function(ax[0,0], x1,x2,pre_1); ax[0,0].set_title('Preactivation')
  draw_2D_function(ax[0,1], x1,x2,pre_2); ax[0,1].set_title('Preactivation')
  draw_2D_function(ax[0,2], x1,x2,pre_3); ax[0,2].set_title('Preactivation')
  draw_2D_function(ax[1,0], x1,x2,act_1); ax[1,0].set_title('Activation')
  draw_2D_function(ax[1,1], x1,x2,act_2); ax[1,1].set_title('Activation')
  draw_2D_function(ax[1,2], x1,x2,act_3); ax[1,2].set_title('Activation')
  draw_2D_function(ax[2,0], x1,x2,w_act_1); ax[2,0].set_title('Weighted Act')
  draw_2D_function(ax[2,1], x1,x2,w_act_2); ax[2,1].set_title('Weighted Act')
  draw_2D_function(ax[2,2], x1,x2,w_act_3); ax[2,2].set_title('Weighted Act')
  plt.show()

  fig, ax = plt.subplots()
  draw_2D_function(ax,x1,x2,y)
  ax.set_title('Network output, $y$')
  ax.set_aspect(1.0)
  plt.show()
```

**`draw_2D_function(ax, x1_mesh, x2_mesh, y)`** dibuja una función escalar definida sobre una cuadrícula bidimensional:

- `ax.contourf(..., levels=256, cmap='hot', vmin=-10, vmax=10)` rellena el plano con colores según el valor de `y`.
- `ax.contour(..., levels=np.arange(-10,10,1.0), cmap='winter')` añade curvas de nivel: puntos con el mismo valor de salida.
- `x1_mesh`, `x2_mesh` y `y` deben tener formas compatibles (en el ejemplo, matrices de `100×100`).

**`plot_neural_2_inputs`** crea primero una cuadrícula `3×3` de mapas: fila 1 = preactivaciones; fila 2 = activaciones; fila 3 = contribuciones ponderadas. Después dibuja un mapa independiente para la salida total `y`. **No hay `TODO`** en esta celda: el notebook pide leerla, no modificarla.

## Celda 4 — ReLU ya resuelta

**Código original — celda 4:**

```python
# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation
```

Aquí el autor utiliza directamente `preactivation.clip(0.0)`. La operación recorta a cero los valores negativos de **cada celda de la malla**. A diferencia del notebook 3.1, este bloque ya está resuelto.

## Celda 5 — Implementar `shallow_2_1_3` (tres `TODO`)

**Código original — celda 5:**

```python
# Define a shallow neural network with, two input, one output, and three hidden units
def shallow_2_1_3(x1,x2, activation_fn, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11,\
                  theta_12, theta_20, theta_21, theta_22, theta_30, theta_31, theta_32):
  # TODO Replace the lines below to compute the three initial linear functions
  # (figure 3.8a-c) from the theta parameters.  These are the preactivations
  pre_1 = np.zeros_like(x1)
  pre_2 = np.zeros_like(x1)
  pre_3 = np.zeros_like(x1)

  # Pass these through the ReLU function to compute the activations as in
  # figure 3.8 d-f
  act_1 = activation_fn(pre_1)
  act_2 = activation_fn(pre_2)
  act_3 = activation_fn(pre_3)

  # TODO Replace the code below to weight the activations using phi1, phi2 and phi3
  # To create the equivalent of figure 3.8 g-i
  w_act_1 = np.zeros_like(x1)
  w_act_2 = np.zeros_like(x1)
  w_act_3 = np.zeros_like(x1)

  # TODO Replace the code below to combing the weighted activations and add
  # phi_0 to create the output as in figure 3.8j
  y = np.zeros_like(x1)

  # Return everything we have calculated
  return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3
```

**Arquitectura:** dos entradas, tres unidades ocultas y una salida. La preactivación de la neurona $i$ es un plano afín:

$$z_i(x_1,x_2)=\theta_{i0}+\theta_{i1}x_1+\theta_{i2}x_2.$$

**Solución propuesta para los tres bloques:**

```python
# 1) Preactivaciones
pre_1 = theta_10 + theta_11*x1 + theta_12*x2
pre_2 = theta_20 + theta_21*x1 + theta_22*x2
pre_3 = theta_30 + theta_31*x1 + theta_32*x2

# 2) Activaciones (el original ya las calcula)
act_1 = activation_fn(pre_1)
act_2 = activation_fn(pre_2)
act_3 = activation_fn(pre_3)

# 3) Contribuciones ponderadas
w_act_1 = phi_1*act_1
w_act_2 = phi_2*act_2
w_act_3 = phi_3*act_3

# 4) Salida
y = phi_0 + w_act_1 + w_act_2 + w_act_3
```

**Interpretación geométrica:** la frontera de activación de cada neurona es la recta $\theta_{i0}+\theta_{i1}x_1+\theta_{i2}x_2=0$. A cada lado de esa recta, ReLU aplica una regla distinta. La combinación produce una superficie lineal por regiones (más exactamente, una función **afín por regiones**).

## Celda 6 — Parámetros, `meshgrid` y ejecución

**Código original — celda 6:**

```python
# Now lets define some parameters and run the neural network
theta_10 =  -4.0 ;  theta_11 = 0.9; theta_12 = 0.0
theta_20 =  5.0  ; theta_21 = -0.9 ; theta_22 = -0.5
theta_30 =  -7  ; theta_31 = 0.5; theta_32 = 0.9
phi_0 = 0.0; phi_1 = -2.0; phi_2 = 2.0; phi_3 = 1.5

x1 = np.arange(0.0, 10.0, 0.1)
x2 = np.arange(0.0, 10.0, 0.1)
x1,x2 = np.meshgrid(x1,x2)  # https://www.geeksforgeeks.org/numpy-meshgrid-function/

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_2_1_3(x1,x2, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_12, theta_20, theta_21, theta_22, theta_30, theta_31, theta_32)
# And then plot it
plot_neural_2_inputs(x1,x2, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3)
```

Los parámetros concretos definen:

$$z_1=-4+0.9x_1,\quad z_2=5-0.9x_1-0.5x_2,\quad z_3=-7+0.5x_1+0.9x_2.$$

Con $\phi_0=0$, $\phi_1=-2$, $\phi_2=2$ y $\phi_3=1.5$:

$$y=-2\operatorname{ReLU}(z_1)+2\operatorname{ReLU}(z_2)+1.5\operatorname{ReLU}(z_3).$$

- Cada `np.arange(0.0,10.0,0.1)` produce 100 valores.
- `np.meshgrid(x1,x2)` convierte los dos vectores en dos matrices de `100×100`: una contiene las coordenadas $x_1$ de cada punto; la otra, sus coordenadas $x_2$.
- Las operaciones vectorizadas evalúan la red en **10 000 puntos** sin un bucle explícito.
- `plot_neural_2_inputs` muestra cómo tres planos se convierten en activaciones recortadas, luego en contribuciones ponderadas y finalmente en la superficie de salida.

## Celda 7 — Pregunta sobre politopos / regiones lineales

El enunciado pregunta cuántas regiones lineales distintas genera el modelo y solicita identificarlas en la salida. Las tres fronteras de activación son:

$$x_1=4/0.9\approx4.444,\quad 0.9x_1+0.5x_2=5,\quad 0.5x_1+0.9x_2=7.$$

Dentro del dominio $[0,10]\times[0,10]$, las tres rectas particionan el cuadrado en **7 regiones con patrones de activación distintos** (no las 8 combinaciones binarias posibles). Una manera reproducible de comprobarlo es evaluar la activación de las tres unidades en una malla densa:

```python
grid = np.linspace(0, 10, 1001)
X1, X2 = np.meshgrid(grid, grid)
z1 = -4 + 0.9*X1
z2 = 5 - 0.9*X1 - 0.5*X2
z3 = -7 + 0.5*X1 + 0.9*X2
patterns = np.stack((z1>0, z2>0, z3>0), axis=-1).reshape(-1,3)
print(np.unique(patterns, axis=0))  # Siete patrones distintos
```

**Precisión:** las fronteras mismas son conjuntos donde alguna preactivación vale cero. El recuento anterior corresponde a regiones bidimensionales del dominio de la figura, no a contar cada frontera como región adicional. Para identificarlas visualmente, observar dónde cambia la pendiente o el patrón de color de la superficie.

## Celda 8 — Extender a dos salidas

El notebook introduce $y_1$ e $y_2$ con **la misma capa oculta**, pero diferentes pesos y sesgos de salida. Matemáticamente:

$$y_1=\phi_{10}+\phi_{11}a_1+\phi_{12}a_2+\phi_{13}a_3,$$
$$y_2=\phi_{20}+\phi_{21}a_1+\phi_{22}a_2+\phi_{23}a_3.$$

Las dos salidas comparten las preactivaciones y activaciones; solo cambia la combinación final.

## Celda 9 — `plot_neural_2_inputs_2_outputs`

**Código original — celda 9:**

```python
# Plot the shallow neural network.  We'll assume input in is range [0,10],[0,10] and output [-10,10]
def plot_neural_2_inputs_2_outputs(x1,x2, y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23):

  # Plot intermediate plots if flag set
  fig, ax = plt.subplots(4,3)
  fig.set_size_inches(8.5, 8.5)
  fig.tight_layout(pad=3.0)
  draw_2D_function(ax[0,0], x1,x2,pre_1); ax[0,0].set_title('Preactivation')
  draw_2D_function(ax[0,1], x1,x2,pre_2); ax[0,1].set_title('Preactivation')
  draw_2D_function(ax[0,2], x1,x2,pre_3); ax[0,2].set_title('Preactivation')
  draw_2D_function(ax[1,0], x1,x2,act_1); ax[1,0].set_title('Activation')
  draw_2D_function(ax[1,1], x1,x2,act_2); ax[1,1].set_title('Activation')
  draw_2D_function(ax[1,2], x1,x2,act_3); ax[1,2].set_title('Activation')
  draw_2D_function(ax[2,0], x1,x2,w_act_11); ax[2,0].set_title('Weighted Act 1')
  draw_2D_function(ax[2,1], x1,x2,w_act_12); ax[2,1].set_title('Weighted Act 1')
  draw_2D_function(ax[2,2], x1,x2,w_act_13); ax[2,2].set_title('Weighted Act 1')
  draw_2D_function(ax[3,0], x1,x2,w_act_21); ax[3,0].set_title('Weighted Act 2')
  draw_2D_function(ax[3,1], x1,x2,w_act_22); ax[3,1].set_title('Weighted Act 2')
  draw_2D_function(ax[3,2], x1,x2,w_act_23); ax[3,2].set_title('Weighted Act 2')
  plt.show()

  fig, ax = plt.subplots()
  draw_2D_function(ax,x1,x2,y1)
  ax.set_title('Network output, $y_1$')
  ax.set_aspect(1.0)
  plt.show()

  fig, ax = plt.subplots()
  draw_2D_function(ax,x1,x2,y2)
  ax.set_title('Network output, $y_2$')
  ax.set_aspect(1.0)
  plt.show()
```

La función de dibujo ahora produce una cuadrícula `4×3`: preactivaciones, activaciones, contribuciones para la salida 1 y contribuciones para la salida 2. Después genera **dos mapas independientes**: uno para $y_1$ y otro para $y_2$. No es un ejercicio `TODO`: es una herramienta de visualización para contrastar que ambas salidas usan las mismas unidades ocultas.

## Celda 10 — Implementar `shallow_2_2_3` (`TODO` integral)

**Código original — celda 10:**

```python
# Define a shallow neural network with, two inputs, two outputs, and three hidden units
def shallow_2_2_3(x1,x2, activation_fn, phi_10,phi_11,phi_12,phi_13, phi_20,phi_21,phi_22,phi_23, theta_10, theta_11,\
                  theta_12, theta_20, theta_21, theta_22, theta_30, theta_31, theta_32):

  # TODO -- write this function -- replace the dummy code below
  pre_1 = np.zeros_like(x1)
  pre_2 = np.zeros_like(x1)
  pre_3 = np.zeros_like(x1)
  act_1 = np.zeros_like(x1)
  act_2 = np.zeros_like(x1)
  act_3 = np.zeros_like(x1)
  w_act_11 = np.zeros_like(x1)
  w_act_12 = np.zeros_like(x1)
  w_act_13 = np.zeros_like(x1)
  w_act_21 = np.zeros_like(x1)
  w_act_22 = np.zeros_like(x1)
  w_act_23 = np.zeros_like(x1)
  y1 = np.zeros_like(x1)
  y2 = np.zeros_like(x1)


  # Return everything we have calculated
  return y1,y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23
```

**El código inicial devuelve exclusivamente ceros** para todos los arreglos. Hay que construir la misma capa oculta del ejercicio anterior y dos combinaciones de salida. **Solución propuesta completa:**

```python
def shallow_2_2_3(x1, x2, activation_fn,
                  phi_10, phi_11, phi_12, phi_13,
                  phi_20, phi_21, phi_22, phi_23,
                  theta_10, theta_11, theta_12,
                  theta_20, theta_21, theta_22,
                  theta_30, theta_31, theta_32):
    pre_1 = theta_10 + theta_11*x1 + theta_12*x2
    pre_2 = theta_20 + theta_21*x1 + theta_22*x2
    pre_3 = theta_30 + theta_31*x1 + theta_32*x2

    act_1 = activation_fn(pre_1)
    act_2 = activation_fn(pre_2)
    act_3 = activation_fn(pre_3)

    w_act_11 = phi_11 * act_1
    w_act_12 = phi_12 * act_2
    w_act_13 = phi_13 * act_3
    w_act_21 = phi_21 * act_1
    w_act_22 = phi_22 * act_2
    w_act_23 = phi_23 * act_3

    y1 = phi_10 + w_act_11 + w_act_12 + w_act_13
    y2 = phi_20 + w_act_21 + w_act_22 + w_act_23
    return (y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3,
            w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23)
```

**Punto importante:** `phi_10` y `phi_20` son los sesgos de las salidas 1 y 2; no deben confundirse con `theta_10` y `theta_20`, que son sesgos de neuronas ocultas. La función devuelve 14 objetos porque el trazador necesita ambas salidas, tres preactivaciones, tres activaciones y seis contribuciones ponderadas.

## Celda 11 — Parámetros y resultados de las dos salidas

**Código original — celda 11:**

```python
# Now lets define some parameters and run the neural network
theta_10 =  -4.0 ;  theta_11 = 0.9; theta_12 = 0.0
theta_20 =  5.0  ; theta_21 = -0.9 ; theta_22 = -0.5
theta_30 =  -7  ; theta_31 = 0.5; theta_32 = 0.9
phi_10 = 0.0; phi_11 = -2.0; phi_12 = 2.0; phi_13 = 1.5
phi_20 = -2.0; phi_21 = -1.0; phi_22 = -2.0; phi_23 = 0.8

x1 = np.arange(0.0, 10.0, 0.1)
x2 = np.arange(0.0, 10.0, 0.1)
x1,x2 = np.meshgrid(x1,x2)  # https://www.geeksforgeeks.org/numpy-meshgrid-function/

# We run the neural network for each of these input values
y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23 = \
    shallow_2_2_3(x1,x2, ReLU, phi_10,phi_11,phi_12,phi_13, phi_20,phi_21,phi_22,phi_23, theta_10, theta_11, theta_12, theta_20, theta_21, theta_22, theta_30, theta_31, theta_32)
# And then plot it
plot_neural_2_inputs_2_outputs(x1,x2, y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23)
```

El código reutiliza los mismos nueve parámetros `theta` y define:

- Salida 1: `phi_10=0`, `phi_11=-2`, `phi_12=2`, `phi_13=1.5`.
- Salida 2: `phi_20=-2`, `phi_21=-1`, `phi_22=-2`, `phi_23=0.8`.

Ambas salidas se evalúan en la misma cuadrícula de 10 000 puntos. **Resultado esperado:** dos mapas de salida diferentes, aun cuando los tres mapas de preactivación y los tres de activación sean idénticos entre ambas salidas.

**Pregunta docente:** ¿qué cambiaría si modificáramos `theta_21`? **Respuesta:** podrían cambiar ambas salidas, porque comparten la segunda neurona oculta. ¿Y si modificáramos `phi_22`? Solo cambiaría la segunda salida.

---

# Parte III — Notebook 4.3: *Deep neural networks*

**Objetivo original:** pasar de una red shallow expresada con escalares a una representación **matricial**, componer dos redes y construir una red con **tres capas ocultas** y dimensiones especificadas.

## Celdas 0–3 — Propósito, bibliotecas y ReLU

- **Celda 0:** enlace a Colab.
- **Celda 1:** objetivo explícito: conversión de redes a forma matricial.
- **Celda 2:** importación de NumPy y Matplotlib.
- **Celda 3:** implementación ya completa de ReLU mediante `preactivation.clip(0.0)`, idéntica en comportamiento a la del notebook 3.2.

## Celda 4 — Red shallow ya implementada

**Código original — celda 4:**

```python
# Define a shallow neural network with, one input, one output, and three hidden units
def shallow_1_1_3(x, activation_fn, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):
  # Initial lines
  pre_1 = theta_10 + theta_11 * x
  pre_2 = theta_20 + theta_21 * x
  pre_3 = theta_30 + theta_31 * x
  # Activation functions
  act_1 = activation_fn(pre_1)
  act_2 = activation_fn(pre_2)
  act_3 = activation_fn(pre_3)
  # Weight activations
  w_act_1 = phi_1 * act_1
  w_act_2 = phi_2 * act_2
  w_act_3 = phi_3 * act_3
  # Combine weighted activation and add y offset
  y = phi_0 + w_act_1 + w_act_2 + w_act_3
  # Return everything we have calculated
  return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3
```

Esta celda contiene la versión **resuelta** de `shallow_1_1_3`: tres preactivaciones, tres activaciones, tres contribuciones ponderadas y una suma final. Sirve como **referencia de verdad** para verificar que la versión matricial produce exactamente la misma función.

## Celdas 5–7 — Graficar y ejecutar la primera red

**Código original — celda 5:**

```python
# # Plot the shallow neural network.  We'll assume input in is range [-1,1] and output [-1,1]
def plot_neural(x, y):
  fig, ax = plt.subplots()
  ax.plot(x.T,y.T)
  ax.set_xlabel('Input'); ax.set_ylabel('Output')
  ax.set_xlim([-1,1]);ax.set_ylim([-1,1])
  ax.set_aspect(1.0)
  plt.show()
```

**Código original — celda 7:**

```python
# Now lets define some parameters and run the first neural network
n1_theta_10 = 0.0   ; n1_theta_11 = -1.0
n1_theta_20 = 0     ; n1_theta_21 = 1.0
n1_theta_30 = -0.67 ; n1_theta_31 =  1.0
n1_phi_0 = 1.0; n1_phi_1 = -2.0; n1_phi_2 = -3.0; n1_phi_3 = 9.3

# Define a range of input values
n1_in = np.arange(-1,1,0.01).reshape([1,-1])

# We run the neural network for each of these input values
n1_out, *_ = shallow_1_1_3(n1_in, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11, n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31)
# And then plot it
plot_neural(n1_in, n1_out)
```

La función `plot_neural` dibuja entrada frente a salida con límites `[−1,1]` en ambos ejes. La primera red usa parámetros con prefijo `n1_`:

$$z_1=-x,\quad z_2=x,\quad z_3=-0.67+x,$$
$$f_1(x)=1-2\operatorname{ReLU}(-x)-3\operatorname{ReLU}(x)+9.3\operatorname{ReLU}(-0.67+x).$$

`np.arange(-1,1,0.01).reshape([1,-1])` crea **200 ejemplos** en una matriz de forma `(1,200)`: una fila de características y una columna por observación. `n1_out, *_ = ...` conserva solo la salida final.

## Celdas 8–9 — Convertir la primera red a matrices (`TODO`)

**Código original — celda 9:**

```python
beta_0 = np.zeros((3,1))
Omega_0 = np.zeros((3,1))
beta_1 = np.zeros((1,1))
Omega_1 = np.zeros((1,3))

# TODO Fill in the values of the beta and Omega matrices with the n1_theta and n1_phi parameters that define the network above
# !!! NOTE THAT MATRICES ARE CONVENTIONALLY INDEXED WITH a_11 IN THE TOP LEFT CORNER, BUT NDARRAYS START AT [0,0]
# To get you started I've filled in a couple:
beta_0[0,0] = n1_theta_10
Omega_0[0,0] = n1_theta_11

# Make sure that input data matrix has different inputs in its columns
n_data = n1_in.size
n_dim_in = 1
n1_in_mat = np.reshape(n1_in,(n_dim_in,n_data))

# This runs the network for ALL of the inputs, x at once so we can draw graph
h1 = ReLU(beta_0 + np.matmul(Omega_0,n1_in_mat))
n1_out = beta_1 + np.matmul(Omega_1,h1)

# Draw the network and check that it looks the same as the non-matrix case
plot_neural(n1_in, n1_out)
```

### Convención de dimensiones

La entrada tiene forma `(1,N)`. Con tres neuronas ocultas:

- `beta_0`: `(3,1)` — sesgos ocultos.
- `Omega_0`: `(3,1)` — pesos de entrada.
- `beta_1`: `(1,1)` — sesgo de salida.
- `Omega_1`: `(1,3)` — pesos de salida.

La red es:

$$H_1=\operatorname{ReLU}(\beta_0+\Omega_0X),\qquad Y=\beta_1+\Omega_1H_1.$$

**Solución propuesta del `TODO`:**

```python
beta_0 = np.array([[n1_theta_10], [n1_theta_20], [n1_theta_30]])
Omega_0 = np.array([[n1_theta_11], [n1_theta_21], [n1_theta_31]])
beta_1 = np.array([[n1_phi_0]])
Omega_1 = np.array([[n1_phi_1, n1_phi_2, n1_phi_3]])
```

**Valores concretos:**

```python
beta_0 = np.array([[0.0], [0.0], [-0.67]])
Omega_0 = np.array([[-1.0], [1.0], [1.0]])
beta_1 = np.array([[1.0]])
Omega_1 = np.array([[-2.0, -3.0, 9.3]])
```

**Qué hacen las instrucciones siguientes:** `np.reshape(n1_in,(1,n_data))` garantiza columnas de observaciones; `np.matmul(Omega_0,n1_in_mat)` calcula todas las preactivaciones ocultas de una vez; NumPy **difunde** `beta_0` desde `(3,1)` a `(3,N)` (broadcasting); `np.matmul(Omega_1,h1)` combina las tres activaciones en una salida para cada columna.

**Verificación más fuerte que comparar gráficos:**

```python
# Conservar antes la salida escalar como y_scalar
h1 = ReLU(beta_0 + Omega_0 @ n1_in_mat)
y_matrix = beta_1 + Omega_1 @ h1
print(np.allclose(y_scalar, y_matrix))  # True
```

Para ejecutar la comprobación anterior, guardar primero `y_scalar = n1_out.copy()` **antes** de sobrescribir `n1_out` en la celda matricial.

## Celdas 10–11 — Segunda red y composición escalar

**Código original — celda 11:**

```python
# Now lets define some parameters and run the second neural network
n2_theta_10 =  -0.6 ; n2_theta_11 = -1.0
n2_theta_20 =  0.2  ; n2_theta_21 = 1.0
n2_theta_30 =  -0.5  ; n2_theta_31 =  1.0
n2_phi_0 = 0.5; n2_phi_1 = -1.0; n2_phi_2 = -1.5; n2_phi_3 = 2.0

# Define a range of input values
n2_in = np.arange(-1,1,0.01)

# We run the second neural network on the output of the first network
n2_out, *_ = \
    shallow_1_1_3(n1_out, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11, n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31)
# And then plot it
plot_neural(n1_in, n2_out)
```

La segunda red recibe **la salida de la primera**, no los valores originales de `n1_in`. Sus preactivaciones son:

$$u_1=-0.6-f_1(x),\quad u_2=0.2+f_1(x),\quad u_3=-0.5+f_1(x),$$

con salida:

$$f_2(f_1(x))=0.5-\operatorname{ReLU}(u_1)-1.5\operatorname{ReLU}(u_2)+2\operatorname{ReLU}(u_3).$$

**Detalle de código:** se crea `n2_in=np.arange(-1,1,0.01)`, pero la llamada efectiva utiliza `n1_out` como entrada de la segunda red. La variable `n2_in` no participa en esa evaluación.

## Celda 12 — Matriz de la red compuesta (`TODO`)

**Código original — celda 12:**

```python
beta_0 = np.zeros((3,1))
Omega_0 = np.zeros((3,1))
beta_1 = np.zeros((3,1))
Omega_1 = np.zeros((3,3))
beta_2 = np.zeros((1,1))
Omega_2 = np.zeros((1,3))

# TODO Fill in the values of the beta and Omega matrices for the n1_theta, n1_phi, n2_theta, and n2_phi parameters
# that define the composition of the two networks above (see eqn 4.5 for Omega1 and beta1 albeit in different notation)
# !!! NOTE THAT MATRICES ARE CONVENTIONALLY INDEXED WITH a_11 IN THE TOP LEFT CORNER, BUT NDARRAYS START AT [0,0] SO EVERYTHING IS OFFSET
# To get you started I've filled in a few:
beta_0[0,0] = n1_theta_10
Omega_0[0,0] = n1_theta_11
beta_1[0,0] = n2_theta_10 + n2_theta_11 * n1_phi_0
Omega_1[0,0] = n2_theta_11 * n1_phi_1


# Make sure that input data matrix has different inputs in its columns
n_data = n1_in.size
n_dim_in = 1
n1_in_mat = np.reshape(n1_in,(n_dim_in,n_data))

# This runs the network for ALL of the inputs, x at once so we can draw graph (hence extra np.ones term)
h1 = ReLU(beta_0 + np.matmul(Omega_0,n1_in_mat))
h2 = ReLU(beta_1 + np.matmul(Omega_1,h1))
n1_out = beta_2 + np.matmul(Omega_2,h2)

# Draw the network and check that it looks the same as the non-matrix version
plot_neural(n1_in, n1_out)
```

### Por qué aparece una matriz `Omega_1` de tamaño 3×3

La primera red tiene tres activaciones $h_1,h_2,h_3$ y una salida escalar intermedia:

$$f_1(x)=\phi_0^{(1)}+\sum_{j=1}^3\phi_j^{(1)}h_j.$$

Cada preactivación de la segunda red es:

$$u_i=\theta_{i0}^{(2)}+\theta_{i1}^{(2)}f_1(x)$$
$$=\underbrace{\theta_{i0}^{(2)}+\theta_{i1}^{(2)}\phi_0^{(1)}}_{\beta_{1,i}}+\sum_{j=1}^3\underbrace{\theta_{i1}^{(2)}\phi_j^{(1)}}_{\Omega_{1,ij}}h_j.$$

Así se **absorbe la salida afín de la primera red en la preactivación de la segunda**: no se aplica una ReLU adicional entre ambas. Por eso la arquitectura matricial resultante es `1 → 3 → 3 → 1`.

**Solución propuesta completa:**

```python
beta_0 = np.array([[n1_theta_10], [n1_theta_20], [n1_theta_30]])
Omega_0 = np.array([[n1_theta_11], [n1_theta_21], [n1_theta_31]])

n2_bias = np.array([[n2_theta_10], [n2_theta_20], [n2_theta_30]])
n2_weight = np.array([[n2_theta_11], [n2_theta_21], [n2_theta_31]])
n1_output_weight = np.array([[n1_phi_1, n1_phi_2, n1_phi_3]])

beta_1 = n2_bias + n2_weight * n1_phi_0
Omega_1 = n2_weight @ n1_output_weight
beta_2 = np.array([[n2_phi_0]])
Omega_2 = np.array([[n2_phi_1, n2_phi_2, n2_phi_3]])
```

**Valores numéricos que deben obtenerse:**

```python
beta_0 = np.array([[0.0], [0.0], [-0.67]])
Omega_0 = np.array([[-1.0], [1.0], [1.0]])
beta_1 = np.array([[-1.6], [1.2], [-1.5]])
Omega_1 = np.array([[2.0, 3.0, -9.3],
                    [-2.0, -3.0, 9.3],
                    [-2.0, -3.0, 9.3]])
beta_2 = np.array([[0.5]])
Omega_2 = np.array([[-1.0, -1.5, 2.0]])
```

**Dimensiones de ejecución:** `h1=(3,N)`, `h2=(3,N)`, `y=(1,N)`. El comentario del notebook menciona un término extra de unos; en el código mostrado **no se construye** un `np.ones`: el sesgo se suma mediante **broadcasting** de NumPy. Esta es la operación efectiva que conviene enseñar.

**Verificación recomendada:** antes de ejecutar la celda 12, guardar `y_composed_scalar = n2_out.copy()` y después comparar con la salida matricial:

```python
h1 = ReLU(beta_0 + Omega_0 @ n1_in_mat)
h2 = ReLU(beta_1 + Omega_1 @ h1)
y_composed_matrix = beta_2 + Omega_2 @ h2
print(np.allclose(y_composed_scalar, y_composed_matrix))  # True
```

## Celdas 13–14 — Red profunda 4 → 5 → 2 → 4 → 1 (`TODO` de dimensiones)

El enunciado define cuatro entradas, tres capas ocultas de 5, 2 y 4 neuronas, y una salida. En la celda 14, `n_data=4` y `x=np.random.normal(size=(D_i,n_data))` producen una matriz de entradas `(4,4)`: cuatro variables por cuatro ejemplos. **No confundir** `D_i=4` (características) con `n_data=4` (observaciones).

**Código original — celda 14:**

```python
# define sizes
D_i=4; D_1=5; D_2=2; D_3=4; D_o=1
# We'll choose the inputs and parameters of this network randomly using np.random.normal
# For example, we'll set the input using
n_data = 4;
x = np.random.normal(size=(D_i, n_data))
# TODO initialize the parameters randomly with the correct sizes
# Replace the lines below
beta_0 = np.random.normal(size=(1,1))
Omega_0 = np.random.normal(size=(1,1))
beta_1 = np.random.normal(size=(1,1))
Omega_1 = np.random.normal(size=(1,1))
beta_2 = np.random.normal(size=(1,1))
Omega_2 = np.random.normal(size=(1,1))
beta_3 = np.random.normal(size=(1,1))
Omega_3 = np.random.normal(size=(1,1))


# If you set the parameters to the correct sizes, the following code will run
h1 = ReLU(beta_0 + np.matmul(Omega_0,x));
h2 = ReLU(beta_1 + np.matmul(Omega_1,h1));
h3 = ReLU(beta_2 + np.matmul(Omega_2,h2));
y = beta_3 + np.matmul(Omega_3,h3)

if h1.shape[0] is not D_1 or h1.shape[1] is not n_data:
  print("h1 is wrong shape")
if h2.shape[0] is not D_2 or h1.shape[1] is not n_data:
  print("h2 is wrong shape")
if h3.shape[0] is not D_3 or h1.shape[1] is not n_data:
  print("h3 is wrong shape")
if y.shape[0] is not D_o or h1.shape[1] is not n_data:
  print("Output is wrong shape")

# Print the inputs and outputs
print("Input data points")
print(x)
print ("Output data points")
print(y)
```

### Derivar las dimensiones sin adivinarlas

Para una capa que transforma $D_{prev}$ entradas en $D_{next}$ neuronas:

$$\Omega\in\mathbb{R}^{D_{next}\times D_{prev}},\qquad\beta\in\mathbb{R}^{D_{next}\times1}.$$

| Objeto | Dimensión correcta | Razón |
|---|---|---|
| `x` | `(4,4)` | 4 entradas × 4 ejemplos |
| `Omega_0` | `(5,4)` | 4 → 5 |
| `beta_0` | `(5,1)` | 5 sesgos |
| `h1` | `(5,4)` | 5 neuronas × 4 ejemplos |
| `Omega_1` | `(2,5)` | 5 → 2 |
| `beta_1` | `(2,1)` | 2 sesgos |
| `h2` | `(2,4)` | 2 neuronas × 4 ejemplos |
| `Omega_2` | `(4,2)` | 2 → 4 |
| `beta_2` | `(4,1)` | 4 sesgos |
| `h3` | `(4,4)` | 4 neuronas × 4 ejemplos |
| `Omega_3` | `(1,4)` | 4 → 1 |
| `beta_3` | `(1,1)` | 1 sesgo |
| `y` | `(1,4)` | 1 salida × 4 ejemplos |

**Solución propuesta del `TODO`:**

```python
beta_0 = np.random.normal(size=(D_1, 1))
Omega_0 = np.random.normal(size=(D_1, D_i))
beta_1 = np.random.normal(size=(D_2, 1))
Omega_1 = np.random.normal(size=(D_2, D_1))
beta_2 = np.random.normal(size=(D_3, 1))
Omega_2 = np.random.normal(size=(D_3, D_2))
beta_3 = np.random.normal(size=(D_o, 1))
Omega_3 = np.random.normal(size=(D_o, D_3))
```

La propagación hacia delante de la celda original queda:

```python
h1 = ReLU(beta_0 + Omega_0 @ x)
h2 = ReLU(beta_1 + Omega_1 @ h1)
h3 = ReLU(beta_2 + Omega_2 @ h2)
y  = beta_3 + Omega_3 @ h3

assert h1.shape == (D_1, n_data)
assert h2.shape == (D_2, n_data)
assert h3.shape == (D_3, n_data)
assert y.shape == (D_o, n_data)
print('Formas correctas:', h1.shape, h2.shape, h3.shape, y.shape)
```

**Detalle del código original:** sus comprobaciones de forma usan `is not` para comparar números enteros, y en varias líneas consultan `h1.shape[1]` incluso cuando se verifica `h2`, `h3` o `y`. Para comprobar dimensiones, es más claro y robusto usar `!=` o, mejor, comparar la tupla completa con `assert`, como en el ejemplo anterior.

**Número total de parámetros:**

$$P=5(4+1)+2(5+1)+4(2+1)+1(4+1)=25+12+12+5=\boxed{54}.$$

**Interpretación:** el notebook inicializa parámetros **aleatorios**, pero no entrena la red. Las salidas impresas son resultados de propagación hacia delante con esos valores aleatorios; no representan predicciones ya ajustadas a un conjunto de datos.

---

# Síntesis transversal: qué aprende el estudiante en cada notebook

| Aspecto | Notebook 3.1 | Notebook 3.2 | Notebook 4.3 |
|---|---|---|---|
| Entradas | 1D | 2D | 1D y luego 4D |
| Salidas | 1 | 1 y luego 2 | 1 |
| Representación | Rectas y función por tramos | Mapas de calor y regiones | Matrices y composición |
| `TODO` centrales | ReLU, preactivaciones, pesos, salida, pérdida, experimentos | Preactivaciones, pesos, salidas y dos salidas | Matrices de red shallow, composición y dimensiones de red profunda |
| Pregunta conceptual | ¿Cómo construye una red una función? | ¿Cómo se divide el plano en regiones? | ¿Cómo representar y componer capas eficientemente? |

## Lista de verificación antes de dar por terminado el laboratorio

- [ ] La ReLU del notebook 3.1 conserva los positivos y anula los negativos.
- [ ] La red `shallow_1_1_3` produce preactivaciones, activaciones y salida no constantes.
- [ ] Los paneles del notebook 3.1 muestran los puntos de quiebre esperados.
- [ ] La pérdida inicial del notebook 3.1 se aproxima a **9.385**.
- [ ] Se responden y comprueban los ocho experimentos sobre parámetros.
- [ ] En el notebook 3.2, `meshgrid` produce arreglos de forma `(100,100)`.
- [ ] Se distinguen siete regiones de activación en el dominio graficado del ejemplo.
- [ ] Las dos salidas del notebook 3.2 comparten activaciones, pero tienen mapas distintos.
- [ ] En el notebook 4.3, la salida matricial coincide con la implementación escalar.
- [ ] La composición de dos redes coincide con la representación matricial `1→3→3→1`.
- [ ] La red `4→5→2→4→1` produce formas `(5,4)`, `(2,4)`, `(4,4)` y `(1,4)`.
- [ ] El estudiante distingue **propagación hacia delante** de **entrenamiento**.

## Preguntas para discusión y evaluación

1. ¿Qué diferencia hay entre una preactivación negativa y una activación ReLU nula?
2. ¿Por qué la red 3.1 puede producir segmentos planos y segmentos con pendiente distinta de cero?
3. ¿Qué ocurre si se cambia un parámetro `theta` frente a un parámetro `phi`?
4. ¿Por qué una escala positiva aplicada antes de ReLU puede compensarse en el peso de salida, pero una escala negativa no?
5. ¿Por qué `np.meshgrid` permite representar una función de dos variables sin bucles anidados explícitos?
6. ¿Qué información proporciona una curva de nivel que no se aprecia inmediatamente en una tabla de valores?
7. ¿Cómo puede una red con una capa oculta compartida producir dos salidas diferentes?
8. ¿Qué significa que los ejemplos se organicen en columnas de una matriz de entrada?
9. ¿Por qué las matrices de la composición no son simplemente la concatenación de las matrices de las dos redes?
10. ¿Por qué tener parámetros aleatorios no significa que una red haya aprendido?

## Bibliografía y trazabilidad

- **Prince, Simon J. D.** (2023). *Understanding Deep Learning*. MIT Press. Capítulo 3 (redes shallow; figuras 3.3 y 3.8) y capítulo 4 (redes profundas; ecuaciones 4.5 y 4.15, figura 4.6).
- **Fuentes primarias de esta guía:** `3_1_Shallow_Networks_I.ipynb` (15 celdas), `3_2_Shallow_Networks_II.ipynb` (12 celdas), `4_3_Deep_Networks.ipynb` (15 celdas), extraídos del archivo adjunto `Notebooks.zip`.
- Los fragmentos titulados **«Código original»** proceden de esas celdas. Las **soluciones propuestas, observaciones docentes, verificaciones y aclaraciones** son elaboración explicativa adicional.
- Repositorio oficial de referencia: https://github.com/udlbook/udlbook/tree/main/Notebooks

