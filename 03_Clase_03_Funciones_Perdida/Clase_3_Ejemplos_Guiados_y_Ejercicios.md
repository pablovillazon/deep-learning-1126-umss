# Clase 3 — Ejemplos guiados y ejercicios

**Módulo:** Deep Learning · **Texto base:** Simon J. D. Prince, *Understanding Deep Learning* (2023), capítulo 5, pp. 56–72.  
**Duración:** 45 minutos · **Requisitos:** logaritmos naturales, sumatorias, nociones de probabilidad y NumPy.

## Propósitos de aprendizaje

1. Distinguir predicción, etiqueta, residuo y función de pérdida.
2. Justificar por qué la máxima verosimilitud conduce a minimizar la log-verosimilitud negativa (NLL).
3. Calcular MSE, BCE y NLL categórica manualmente y con NumPy.
4. Interpretar cómo influye la confianza de un clasificador en la pérdida.

## Secuencia de trabajo (45 min)

| Minutos | Actividad | Método |
|---|---|---|
| 0–5 | Recordatorio: salida de una red y elección de pérdida | Preguntas rápidas |
| 5–15 | Ejemplo 1: regresión Gaussiana y MSE | Guiado |
| 15–25 | Ejemplo 2: Bernoulli y BCE | Guiado |
| 25–35 | Ejemplo 3: softmax y NLL | Guiado |
| 35–43 | Ejercicios individuales / parejas | Resolución |
| 43–45 | Cierre y discusión | Puesta en común |

---

## Ejemplo guiado 1. ¿Cómo se obtiene MSE?

**Situación.** Un modelo predice un valor real. Tenemos observaciones $y=[2,4,6]$ y predicciones $\hat y=[3,3,5]$.

**Paso 1.** Residuos $r_i=y_i-\hat y_i=[-1,1,1]$.

**Paso 2.** Errores cuadrados $r_i^2=[1,1,1]$.

**Paso 3.** Suma de cuadrados $\mathrm{SSE}=1+1+1=3$.

**Paso 4.** Promedio: $\mathrm{MSE}=\frac13\sum_i(y_i-\hat y_i)^2=1$.

**Relación con el libro (§5.3).** Si suponemos $y_i|x_i\sim\mathcal N(\mu_i,\sigma^2)$ y $\mu_i=f(x_i;\phi)$, entonces

$$
-\log p(D|\phi)=\frac{N}{2}\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\sum_i(y_i-\mu_i)^2.
$$

Para $\sigma^2$ constante respecto de $\phi$, minimizar NLL equivale a minimizar SSE (y, con $N$ fijo, MSE). **Advertencia:** NLL y MSE no son idénticas numéricamente; difieren en escala y un término constante.

```python
import numpy as np
y = np.array([2., 4., 6.])
y_hat = np.array([3., 3., 5.])
print('SSE:', np.sum((y-y_hat)**2))
print('MSE:', np.mean((y-y_hat)**2))
```

**Pregunta de discusión.** ¿Qué ocurriría con MSE si un único error pasara de 1 a 10?

---

## Ejemplo guiado 2. Clasificación binaria: de Bernoulli a BCE

**Situación.** Una red estima la probabilidad de la clase positiva $p=P(y=1|x)$. La etiqueta real es $y=1$ y la red produce $p=0.8$.

**Modelo probabilístico (§5.4):** $P(y|p)=p^y(1-p)^{1-y}$.

**Aplicar NLL:**

$$
\ell(y,p)=-\log P(y|p)=-[y\log p+(1-y)\log(1-p)].
$$

**Cálculo:** $\ell(1,0.8)=-\log(0.8)\approx0.2231$.

**Contraste:** Si la red dijera $p=0.1$, $\ell(1,0.1)=-\log(0.1)\approx2.3026$. Una predicción equivocada y muy confiada es mucho más penalizada.

**Para $y=0$:** la pérdida es $-\log(1-p)$, por ejemplo con $p=0.2$ vale $0.2231$.

```python
import numpy as np

def bce(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

print(bce(np.array([1]), np.array([0.8])))
```

**Pregunta de discusión.** ¿Por qué no basta con medir el porcentaje de aciertos?

---

## Ejemplo guiado 3. Softmax y NLL multiclase

**Situación.** Una red de tres clases produce logits $z=[2,1,0]$. La etiqueta verdadera es la primera clase (índice 0 en Python).

**Paso 1.** Exponenciales: $e^z\approx[7.3891,2.7183,1]$.

**Paso 2.** Normalización: $\sum_j e^{z_j}\approx11.1074$.

**Paso 3.** Softmax: $p\approx[0.6652,0.2447,0.0900]$.

**Paso 4.** Pérdida categórica: $\ell=-\log p_{\mathrm{verdadera}}\approx-\log(0.6652)=0.4076$.

**Estabilidad numérica.** Restar $m=\max_j z_j$ antes de exponenciar produce exactamente las mismas probabilidades y evita desbordamientos.

```python
import numpy as np
z = np.array([2., 1., 0.])
z_stable = z - np.max(z)
p = np.exp(z_stable) / np.exp(z_stable).sum()
print('Probabilidades:', p)
print('NLL:', -np.log(p[0]))
```

**Pregunta de discusión.** ¿Qué cambia si la clase verdadera es la tercera (índice 2)?

---

## Ejercicios para estudiantes

### E1. Comparación de regresores

Valores reales $y=[1,3,5]$. Modelo A: $\hat y_A=[1,4,4]$; modelo B: $\hat y_B=[2,2,5]$. Calcular SSE y MSE. ¿Existe un ganador bajo MSE?

### E2. BCE promedio

Etiquetas $y=[1,0,1]$ y probabilidades $p=[0.9,0.2,0.6]$. Calcular BCE de cada observación y BCE promedio. Identificar la predicción más penalizada.

### E3. NLL categórica

Un clasificador entrega $p=[0.2,0.5,0.3]$. La etiqueta correcta es la tercera clase. Calcular la NLL y explicar qué pasa cuando la probabilidad de la clase correcta tiende a cero.

### E4. Diseño de una pérdida

Relacionar cada tarea con una distribución y pérdida: (a) precio de vivienda, (b) correo spam/no spam, (c) dígito del 0 al 9. Justificar cada decisión usando el dominio de la salida.

### E5. Precisión frente a pérdida

Con etiquetas $y=[1,0]$, comparar predicciones $p_A=[0.51,0.49]$ y $p_B=[0.99,0.01]$. ¿Tienen la misma accuracy? ¿La misma BCE?

---

## Solucionario docente (no proyectar inicialmente)

**E1.** A: cuadrados $[0,1,1]$ → SSE = 2, MSE = $2/3$. B: cuadrados $[1,1,0]$ → SSE = 2, MSE = $2/3$. Empate bajo MSE.

**E2.** Pérdidas: $-\ln(0.9)=0.10536$, $-\ln(0.8)=0.22314$, $-\ln(0.6)=0.51083$. Media $\approx0.27978$. La tercera observación aporta la mayor pérdida.

**E3.** $-\ln(0.3)\approx1.20397$. Cuando $p_{\mathrm{correcta}}\to0^+$, la pérdida tiende a $+\infty$.

**E4.** (a) Normal + MSE bajo varianza fija; (b) Bernoulli + BCE; (c) categórica + softmax/NLL. La elección depende de las hipótesis sobre los datos.

**E5.** Ambas accuracy = 100 % con umbral 0.5; BCE(A) = $-\ln(0.51)\approx0.67334$, BCE(B) = $-\ln(0.99)\approx0.01005$.

## Preguntas de cierre

- ¿Por qué la función de pérdida no se elige solo por conveniencia computacional?
- ¿Cuál es la relación entre NLL y máxima verosimilitud?
- ¿Por qué BCE y NLL categórica dependen de probabilidades y no solo de la clase predicha?
- ¿Cómo se conecta la pérdida con el entrenamiento que veremos en la clase 4?

**Referencia:** Prince (2023), *Understanding Deep Learning*, capítulo 5, §§5.1–5.7. Los ejemplos y ejercicios son elaboraciones docentes propias.
