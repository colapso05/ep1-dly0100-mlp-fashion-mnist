# EP1 — Deep Learning (DLY0100)
## Clasificación de prendas de vestir con una Red Neuronal Multicapa (MLP)

**Asignatura:** Deep Learning (DLY0100) · **Evaluación:** Parcial N°1 (30%)
**Alumno:** Demis Zuñiga
**Dataset:** [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) (indicado por el docente)
**Framework:** TensorFlow 2.21 / Keras 3.15

---

## Qué contiene este repositorio

El entregable principal es **el cuaderno Jupyter**, que la pauta considera el informe: cada
decisión de diseño, preprocesamiento, arquitectura e hiperparámetro está justificada en celdas
Markdown junto al código que la implementa y a la salida ya ejecutada.

| Archivo | Qué es |
|---|---|
| `notebooks/EP1_DLY0100_FashionMNIST_MLP.ipynb` | **El entregable.** Cuaderno completo, ejecutado, con las 18 figuras y todas las salidas |
| `images/` | Las 18 figuras exportadas en PNG |
| `data/resumen_experimentos.csv` | Resultados de los experimentos controlados |
| `data/tabla_comparativa_configuraciones.csv` | Comparación de todas las configuraciones probadas |
| `requirements.txt` | Dependencias |

---

## Cómo ejecutar el proyecto

### Opción A — Google Colab (recomendada)

1. Entrar a [colab.research.google.com](https://colab.research.google.com) y elegir
   **Archivo → Subir notebook**, seleccionando `notebooks/EP1_DLY0100_FashionMNIST_MLP.ipynb`.
2. En el menú, **Entorno de ejecución → Ejecutar todas**.
3. Fashion-MNIST se descarga automáticamente con `keras.datasets.fashion_mnist.load_data()`
   (~30 MB la primera vez). No hay que subir ningún archivo.

Tiempo total de ejecución: **unos 13 minutos** en CPU.

### Opción B — Local

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook notebooks/EP1_DLY0100_FashionMNIST_MLP.ipynb
```

Luego **Kernel → Restart & Run All**.

El cuaderno localiza la carpeta `data/` automáticamente y crea `images/` y `models/` si no
existen, así que funciona igual abierto desde la raíz del proyecto o desde `notebooks/`.

### Reproducibilidad

Todas las operaciones aleatorias están fijadas con **`SEED = 42`** (NumPy, TensorFlow y Keras), y
la partición de entrenamiento/validación usa esa misma semilla. Dos ejecuciones producen los
mismos números.

---

## El problema

Dada una imagen en escala de grises de 28×28 píxeles de una prenda de vestir, predecir a cuál de
las **10 categorías** pertenece:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Camiseta | Pantalón | Suéter | Vestido | Abrigo | Sandalia | Camisa | Zapatilla | Bolso | Botín |

**Datos:** 60.000 imágenes de entrenamiento y 10.000 de prueba, **perfectamente balanceadas**
(6.000 por clase). El clasificador trivial que siempre responde la misma categoría acierta el
**10%**: ese es el piso de referencia.

---

## Resultados

### Modelo final

Evaluado sobre las **10.000 imágenes de prueba**, que no se usaron en ninguna decisión de diseño:

| Métrica | Valor |
|---|---|
| **Accuracy** | **0,8776** |
| Precision (macro) | 0,8774 |
| Recall (macro) | 0,8776 |
| F1-score (macro) | 0,8753 |
| Parámetros | 109.386 |

**Mejora sobre el modelo base: +1,06 puntos** de exactitud (0,8670 → 0,8776).

### Comparación de las 10 clases (modelo base)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Pantalón | 0,990 | 0,957 | 0,973 |
| Bolso | 0,963 | 0,967 | 0,965 |
| Botín | 0,940 | 0,958 | 0,949 |
| Sandalia | 0,976 | 0,921 | 0,948 |
| Zapatilla | 0,917 | 0,957 | 0,936 |
| Vestido | 0,890 | 0,863 | 0,876 |
| Camiseta | 0,824 | 0,814 | 0,819 |
| Abrigo | 0,767 | 0,784 | 0,775 |
| Suéter | 0,765 | 0,758 | 0,761 |
| **Camisa** | **0,657** | **0,691** | **0,674** |

---

## Metodología: experimentos controlados

La pauta pide analizar el impacto de los parámetros **variando un parámetro a la vez**. En cada
experimento se cambia **una sola cosa** y todo lo demás queda fijo.

### Experimento 1 — Épocas

| Épocas | Accuracy | Pérdida train | Pérdida val |
|---|---|---|---|
| 5 | 0,8558 | 0,3586 | 0,3659 |
| 15 | 0,8670 | 0,2308 | 0,3431 |
| 30 | 0,8670 | 0,1988 | 0,3544 |

Con 5 épocas el modelo está **subajustado**; a partir de 15 ya no mejora, y la parada temprana
cortó en la época 20 de un tope de 30.

### Experimento 2 — Tasa de aprendizaje

| learning_rate | Accuracy | Pérdida val final |
|---|---|---|
| 0,0001 | 0,8472 | 0,3969 |
| **0,001** | **0,8670** | **0,3431** |
| 0,01 | 0,8611 | 0,4023 |

La tasa baja no alcanza a converger; la alta oscila porque se pasa de largo el mínimo.

### Experimento 3 — Tamaño de batch

| batch_size | Accuracy | Pasos por época | Segundos por época |
|---|---|---|---|
| 32 | 0,8596 | 625 | 2,64 |
| **128** | **0,8670** | 157 | 0,86 |
| 512 | 0,8571 | 40 | 0,59 |

Más pasos por época no implican mejor resultado: el batch de 32 hace 4 veces más actualizaciones
y tarda 3 veces más por época, con peor exactitud.

### Comparación de funciones de activación

| Activación | Accuracy | F1 macro | Gradiente capa 1 / capa salida |
|---|---|---|---|
| **ReLU** | **0,8670** | **0,8677** | 3,25 |
| tanh | 0,8630 | 0,8628 | 3,44 |
| sigmoid | 0,8616 | 0,8609 | **0,39** |

La diferencia en exactitud es modesta (medio punto), pero la **medición directa del gradiente**
con `tf.GradientTape` muestra el desvanecimiento con `sigmoid`: el gradiente que llega a la
primera capa es 2,6 veces menor que el que llega a la salida. Con `tanh` no hay desvanecimiento
pero el resultado es peor igual: **la magnitud del gradiente no predice por sí sola el
rendimiento.**

### Comparación de funciones de pérdida

| Pérdida | Accuracy | F1 macro |
|---|---|---|
| **categorical_crossentropy** | **0,8670** | **0,8677** |
| mean_squared_error | 0,8632 | 0,8629 |

Diferencia pequeña en este modelo. Se elige entropía cruzada por corrección del gradiente, que es
proporcional al error, no por un salto de rendimiento medible.

### Regularización (20 épocas)

| Configuración | Accuracy | **Brecha train−val** |
|---|---|---|
| Sin regularización | 0,8670 | 0,1556 |
| **Dropout 0,3** | **0,8670** | **0,0227** |
| Batch Normalization | 0,8533 | 0,2615 |
| Dropout + BatchNorm | 0,8548 | 0,0660 |

**Dropout reduce la brecha casi siete veces** manteniendo la exactitud: eso es generalizar mejor.
Batch Normalization **no aportó** en este modelo (peor exactitud y mayor brecha), y el cuaderno
explica por qué en lugar de omitirlo.

### Optimizadores

| Optimizador | Accuracy |
|---|---|
| **Adam** | **0,8670** |
| RMSprop | 0,8594 |
| SGD | 0,8141 |

**Es el hiperparámetro que más movió el resultado**: más de 5 puntos entre Adam y SGD.

---

## Hallazgos principales

1. **Los errores no son aleatorios.** Se concentran en las categorías visualmente parecidas:
   `Abrigo`-`Camisa` (similitud 0,9874), `Suéter`-`Abrigo` (0,9841) y `Suéter`-`Camisa` (0,9832).
   La hipótesis se planteó antes de entrenar y se verificó al final.
2. **`Camisa` es la clase más difícil** (F1 0,674): se distingue de la camiseta por el cuello y
   los botones, detalles que desaparecen a 28×28 píxeles en escala de grises. No es una falla del
   modelo, es un límite de la información disponible.
3. **La confianza separa aciertos de errores** (0,925 frente a 0,664), lo que habilita un
   **umbral de abstención** en producción: derivar a revisión humana en vez de etiquetar mal.
4. **El optimizador pesa más que la función de pérdida** en este modelo. Saber dónde está el
   rendimiento es tan importante como obtenerlo.

---

## Limitaciones

- Un MLP **no tiene invarianza a la traslación**: cada píxel tiene un peso atado a una posición
  fija, y al aplanar la imagen se pierde la noción de vecindad. Es la motivación de las redes
  convolucionales (EA2).
- La resolución de 28×28 borra los detalles que distinguen camisa, camiseta y suéter.
- Los experimentos exploratorios usaron un subconjunto estratificado de 20.000 imágenes por costo
  computacional; el modelo final sí usó todos los datos.
- **No se aisló el aporte del aumento de datos frente al del Dropout** en el modelo final: ambos
  cambiaron a la vez.
- La escala de grises desaprovecha el color, que es información útil para distinguir prendas.

---

## Mejoras propuestas

1. **Reemplazar el MLP por una CNN**, invariante a la traslación y capaz de explotar la vecindad
   entre píxeles.
2. **Aumento de datos** (desplazamientos, rotaciones leves y volteo horizontal).
3. **Búsqueda sistemática de hiperparámetros** (Keras Tuner) en lugar de la exploración manual.
4. **Monitorear el recall de `Camisa`** en producción, por ser la categoría más problemática.
5. **Umbral de abstención** aprovechando que los errores se concentran en confianzas bajas.
