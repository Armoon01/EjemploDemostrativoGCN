# 🧠 Ejemplo Demostrativo de GCN

## VI. Un Ejemplo Demostrativo de GCN

### a. 📝 Descripción del Problema

Este proyecto presenta una **clasificación de nodos** utilizando redes neuronales convolucionales sobre grafos (**Graph Convolutional Networks, GCN**).  
El objetivo es predecir la etiqueta o clase de cada nodo en un grafo, basado en sus características y su conectividad. Este tipo de problema es común en áreas como análisis de redes sociales, biología computacional y sistemas de recomendación.

---

### b. 📚 Obtención del Dataset

Para este ejemplo se utiliza el dataset **Cora**, un conjunto de datos clásico de citaciones científicas en el ámbito de aprendizaje de máquinas.  
Cada nodo representa un artículo y las aristas indican citaciones entre ellos.  
El dataset puede obtenerse automáticamente usando librerías como [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) o [DGL](https://www.dgl.ai/), por ejemplo:

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')
```

---

### c. ⚙️ Configuración de la GCN

El modelo GCN se configura con las siguientes características principales:

- **Capas:** Dos capas convolucionales sobre grafos.
- **Función de activación:** ReLU.
- **Regularización:** Dropout y weight decay para evitar sobreajuste.
- **Optimizador:** Adam.
- **Entrenamiento:** Se entrena para minimizar la pérdida de clasificación sobre los nodos de entrenamiento.

Ejemplo de definición del modelo (usando PyTorch Geometric):

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

### d. 🤖 Descripción del Algoritmo e Implementación

El algoritmo implementado sigue estos pasos:

1. **Carga del dataset** y conversión a formato adecuado para el modelo.
2. **Definición del modelo GCN** como una secuencia de capas convolucionales sobre grafos.
3. **Entrenamiento:** 
   - Se propagan los datos por la GCN.
   - Se calcula la función de pérdida (cross-entropy).
   - Se actualizan los pesos del modelo mediante backpropagation.
4. **Evaluación:** 
   - Se mide la precisión en el conjunto de prueba (nodos no vistos durante el entrenamiento).

El código está documentado línea por línea para facilitar su comprensión y replicación.

---

### 🖼️ Visualización

A continuación se muestra una imagen de la visualización del grafo y/o los resultados de la clasificación de nodos usando la GCN:

![Visualización de GCN](image.png)

---

## 👨‍💻 Autores

- **Andy José Luna Izaguirre**
- **Alejandro Baires Pérez**
- **Ángel Andrés Rojas Ruano**
- **Aarón José Guevara Mora**

**Grupo:** 04-10am

---

## 📖 Referencias y fuentes

- [Kipf, T.N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.](https://arxiv.org/abs/1609.02907)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [DGL Documentation](https://docs.dgl.ai/)

---

## 🎬 Demo

Incluye un ejemplo demostrativo en el archivo principal del repositorio.  

---

## 📄 Licencia

Este proyecto se entrega con fines educativos.  
Por favor, cita a los autores y las fuentes originales en cualquier reutilización.
