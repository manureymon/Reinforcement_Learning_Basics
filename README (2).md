## Reinforcement Learning Basics

El Aprendizaje por Refuerzo (RL) es una metodología de aprendizaje automático en la que un agente aprende a tomar decisiones mediante la interacción con un entorno. 

# Blackjack Q-Learning Agent
El objetivo de este proyecto es el siguiente:

Entrenaremos un agente de Aprendizaje por Refuerzo (Reinforcement Learning) para jugar Blackjack utilizando un método basado en la ecuación de Bellman. Se implementará una estrategia epsilon-greedy para la selección de acciones y se evaluará el desempeño del modelo después del entrenamiento.

## Método de Bellman para Actualizar los Valores de $Q$

La ecuación de Bellman se usa para actualizar los valores $Q(s, a)$ asociados a cada estado y acción:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

donde:
- $Q(s, a)$ representa el valor de una acción $a$ en un estado $s$.
- $\alpha$ es la tasa de aprendizaje (learning rate).
- $r$ es la recompensa obtenida después de tomar la acción $a$.
- $\gamma$ es el factor de descuento (discount factor), que pondera la importancia de las futuras recompensas.
- $\max_{a'} Q(s', a')$ es el valor estimado de la mejor acción posible en el siguiente estado $s'$.

## Librerías Utilizadas

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
```

## Configuración

El script utiliza los siguientes hiperparámetros:

- alpha (0.4): Tasa de aprendizaje.

- gamma (0.9): Factor de descuento para futuras recompensas.

- epsilon (0.4): Probabilidad de exploración en la política epsilon-greedy.

- train_episodes (250000): Número de episodios de entrenamiento.

- test_episodes (50): Número de episodios de prueba tras el entrenamiento.

## Explicación del Código

1. **Inicialización**: Se configura el entorno de Blackjack usando `gymnasium` y se define la tabla $Q$ como un `defaultdict`.
2. **Política Epsilon-Greedy**: Se implementa una estrategia donde el agente elige una acción aleatoria con probabilidad $\epsilon$ y la mejor acción conocida con $1 - \epsilon$.
3. **Entrenamiento**: Durante 250,000 episodios, el agente juega partidas de Blackjack y actualiza su tabla $Q$ usando la ecuación de Bellman.
4. **Pruebas**: Tras el entrenamiento, el agente juega 50 partidas sin exploración, seleccionando siempre la acción óptima.
5. **Visualización**: Se grafica el desempeño del agente en términos de victorias, derrotas y empates.
6. **Cierre del Entorno**: Se finaliza la simulación cerrando el entorno de `gymnasium`.



## Consideraciones Adicionales

En la vida real, los casinos utilizan múltiples barajas en el Blackjack para dificultar el conteo de cartas. Sería recomendable modificar la simulación para incluir diferentes cantidades de barajas y analizar cómo afecta la estrategia del agente:

* **Stack de 4 Barajas:**
    El agente juega con un stack de 4 barajas de cartas, lo que incrementa la complejidad del entorno y permite al agente aprender estrategias más avanzadas.
* **Contar Cartas:**
    El agente implementa una técnica básica de conteo de cartas (Hi-Lo) para calcular si es favorable pedir una carta adicional. Se utiliza un contador que incrementa con cartas bajas (2-6) y decrementa con cartas altas (10, J, Q, K, A), lo que le ayuda a determinar si las probabilidades están a su favor para pedir otra carta.


by:
- Rania Aguirre
- Manuel Reyna
