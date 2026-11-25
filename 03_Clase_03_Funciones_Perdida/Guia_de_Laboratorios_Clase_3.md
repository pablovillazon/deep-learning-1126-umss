# Guía de laboratorios — Clase 3: Funciones de pérdida

**Texto base:** Simon J. D. Prince, *Understanding Deep Learning* (2023), capítulo 5, §§5.1–5.7.  
**Notebooks:** `Lab_01_Regresion_MSE.ipynb`, `Lab_02_Clasificacion_BCE.ipynb`, `Lab_03_Softmax_NLL.ipynb`.  
**Naturaleza del material:** laboratorios docentes **originales**, creados para acompañar el capítulo 5; no son notebooks oficiales del repositorio del libro.

## 1. Objetivo global

Relacionar tres tipos de tareas predictivas con sus distribuciones probabilísticas y funciones de pérdida. La prioridad es **comprender la matemática antes de ejecutar código** y luego verificar las expresiones mediante NumPy.

## 2. Organización del laboratorio (75 minutos + cierre)

| Tiempo | Trabajo | Evidencia |
|---|---|---|
| 21:25–21:45 | Laboratorio 1: regresión | SSE, MSE, RMSE y NLL |
| 21:45–22:05 | Laboratorio 2: clasificación binaria | Sigmoide, BCE y comparación |
| 22:05–22:30 | Laboratorio 3: clasificación multiclase | Softmax, NLL y estabilidad |
| 22:30–22:40 | Experimentos comparativos | Respuestas justificadas |
| 22:40–23:00 | Discusión y entrega | Notebook ejecutado y conclusiones |

## 3. Preparación del entorno

Se requiere Python 3.10+, Jupyter o Google Colab, NumPy y Matplotlib. Instalar si es necesario:

```bash
python -m pip install numpy matplotlib notebook
```

Ejecutar cada notebook desde el principio. Las celdas `TODO` **no están resueltas intencionalmente**: algunas incluyen `raise NotImplementedError` y requieren intervención del estudiante.

---

# Laboratorio 1 — Regresión: MSE y modelo Gaussiano

## 1.1. Pregunta científica

¿Cuál es la diferencia entre medir el error cuadrático y asumir una distribución normal de las observaciones?

La red predice una media $\mu_i=f(x_i;\phi)$. Bajo $y_i\sim\mathcal N(\mu_i,\sigma^2)$:

$$
\mathrm{NLL}=\frac{N}{2}\ln(2\pi\sigma^2)+\frac{1}{2\sigma^2}\sum_i(y_i-\mu_i)^2.
$$

Si $\sigma^2$ es fija, el orden de los modelos según NLL coincide con el orden según SSE o MSE.

## 1.2. Bloque de datos

```python
import numpy as np
y = np.array([2.,4.,6.,8.])
y_hat = np.array([3.,3.,5.,9.])
```

**Interpretación:** `y` contiene etiquetas reales y `y_hat` predicciones. El error se calcula elemento a elemento mediante difusión de NumPy (broadcasting compatible).

## 1.3. TODO: SSE, MSE y RMSE

La implementación propuesta es:

```python
def sse(y, y_hat):
    return np.sum((y-y_hat)**2)

def mse(y, y_hat):
    return np.mean((y-y_hat)**2)

def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))
```

**Verificación:** errores cuadrados $[1,1,1,1]$; SSE = 4; MSE = 1; RMSE = 1. Los tres valores son distintos conceptualmente aunque coincidan numéricamente en este ejemplo.

## 1.4. Bloque gráfico

El notebook traza observaciones y predicciones. Cada diferencia vertical representa un residuo. Preguntar: ¿cuáles residuos contribuyen más a SSE y por qué?

## 1.5. TODO: NLL Gaussiana

```python
def gaussian_nll(y, mu, sigma):
    if sigma <= 0:
        raise ValueError('sigma debe ser positiva')
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.sum(0.5*np.log(2*np.pi*sigma**2) + (y-mu)**2/(2*sigma**2))
```

**Atención:** `sigma` es la **desviación estándar**, no la varianza. El término $\log(\sigma^2)$ cambia si se modifica `sigma`. Si se fija, la comparación de parámetros $\phi$ queda gobernada por SSE.

## 1.6. Experimento

Comparar dos modelos con MSE y NLL usando el mismo `sigma`. Repetir con distintos `sigma`. Describir qué cambia y qué permanece.

---

# Laboratorio 2 — Clasificación binaria: sigmoide y BCE

## 2.1. Pregunta científica

¿Por qué es importante la **probabilidad** que asigna el modelo a una etiqueta, además de acertar la etiqueta?

La red produce un logit $z\in\mathbb R$; la sigmoide produce $p=1/(1+e^{-z})$. El modelo Bernoulli asigna $P(y|p)=p^y(1-p)^{1-y}$.

## 2.2. Bloque sigmoide

```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
```

Interpretar $\sigma(0)=0.5$, $\sigma(z)\to1$ si $z\to+\infty$ y $\sigma(z)\to0$ si $z\to-\infty$. Para logits extremos, esta versión introductoria puede sufrir desbordamiento intermedio; no se utiliza como implementación industrial.

## 2.3. TODO: BCE

$$
\mathrm{BCE}=-\frac1N\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)].
$$

```python
def binary_cross_entropy(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    y = np.asarray(y, dtype=float)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
```

`np.clip` evita evaluar `log(0)`. Es una aproximación numérica práctica que modifica ligeramente probabilidades extremadamente cercanas a 0 o 1.

**Verificación:** para `y=[1,0,1]`, `p=[0.9,0.2,0.6]`, BCE media $\approx0.27978$.

## 2.4. Bloque visualización

La curva $-\log p$ para $y=1$ decrece a medida que la probabilidad correcta aumenta. Pedir que expliquen por qué una predicción errónea muy segura es fuertemente penalizada.

## 2.5. TODO: accuracy vs BCE

```python
y = np.array([1,0,1,0])
p_a = np.array([.51,.49,.51,.49])
p_b = np.array([.95,.05,.95,.05])
accuracy = lambda y,p: np.mean((p >= .5).astype(int) == y)
print('A:', accuracy(y,p_a), binary_cross_entropy(y,p_a))
print('B:', accuracy(y,p_b), binary_cross_entropy(y,p_b))
```

**Interpretación:** ambas accuracy = 1.0; BCE(A) $\approx0.67334$ y BCE(B) $\approx0.05129$. BCE distingue la confianza incluso cuando accuracy es idéntica.

---

# Laboratorio 3 — Clasificación multiclase: softmax y NLL

## 3.1. Pregunta científica

¿Cómo convertir K salidas arbitrarias de una red en una distribución de probabilidad válida?

Los logits $z_k$ no son probabilidades. La función softmax garantiza $p_k>0$ y $\sum_k p_k=1$.

## 3.2. TODO: softmax estable

```python
def softmax(z):
    z = np.asarray(z, dtype=float)
    z_shift = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z_shift)
    return e / np.sum(e, axis=-1, keepdims=True)
```

`axis=-1` opera sobre la dimensión de clases y funciona para un vector o para lotes de filas. `keepdims=True` conserva dimensiones compatibles para la división.

**Verificación:** para `z=[2,1,0]`, probabilidades aproximadas `[0.66524,0.24473,0.09003]`; su suma debe ser 1.

## 3.3. TODO: NLL categórica

```python
def categorical_nll(probs, target_index):
    probs = np.asarray(probs, dtype=float)
    return -np.log(np.clip(probs[target_index], 1e-12, 1.0))
```

Para clase verdadera de índice 0, $-\ln(0.66524)\approx0.40761$. Si la etiqueta verdadera fuese índice 2, la pérdida aumentaría a $\approx2.40761$.

## 3.4. Bloque de estabilidad numérica

Probar `z=[1000,1001,999]`. La implementación directa `np.exp(z)` puede producir infinito. La versión estable resta el máximo y devuelve probabilidades válidas.

## 3.5. TODO: lotes de observaciones

```python
batch_logits = np.array([[2.,1.,0.],[0.,2.,1.],[1.,0.,3.]])
targets = np.array([0,1,2])
probs = softmax(batch_logits)
chosen = probs[np.arange(len(targets)), targets]
mean_nll = -np.mean(np.log(chosen))
print('Probabilidades:', probs)
print('NLL media:', mean_nll)
```

**Interpretación:** `probs` tiene forma `(3,3)`; `chosen` selecciona la probabilidad de la clase correcta para cada observación. Esta operación vectorizada reemplaza un ciclo explícito.

---

## 4. Evaluación formativa

| Criterio | Logrado | En proceso |
|---|---|---|
| Explica la relación entre MSE y NLL Gaussiana | ☐ | ☐ |
| Implementa SSE, MSE y RMSE sin ciclos | ☐ | ☐ |
| Deriva BCE desde Bernoulli | ☐ | ☐ |
| Compara accuracy y BCE con fundamento | ☐ | ☐ |
| Implementa softmax estable y NLL | ☐ | ☐ |
| Explica los resultados de los tres notebooks | ☐ | ☐ |

**Entrega sugerida:** los tres `.ipynb` ejecutados, con cada `TODO` resuelto, gráficos y una conclusión de 4–6 líneas por laboratorio.

## 5. Preguntas para la discusión final

1. ¿Es lo mismo minimizar MSE que minimizar NLL Gaussiana si la varianza también es un parámetro entrenable?
2. ¿Por qué una probabilidad correcta muy baja genera una pérdida alta?
3. ¿Qué diferencia existe entre clasificación multiclase exclusiva y clasificación multietiqueta?
4. ¿Por qué no debemos calcular softmax ingenuamente con logits grandes?
5. ¿Qué procedimiento emplearemos en la clase 4 para modificar los parámetros y disminuir estas pérdidas?

**Bibliografía:** Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. Capítulo 5. https://udlbook.github.io/udlbook/
