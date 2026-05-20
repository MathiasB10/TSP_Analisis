[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

# Análisis de Algoritmos TSP en Plano Uniforme
Análisis matemático e ingenieríl aplicado en proyecto experimental.

## Autor
Mathias Gabriel Barrezueta Gonzalez

## Resumen del proyecto
Este proyecto demuestra como se comportan 3 algoritmos de optimización para el TSP (Traveling Salesman Problem) a medida que el número de nodos (n) va aumentando desde 5 hasta 50 entre 5 y 5. Para poder realizar el cómputo se generan puntos aleatorios uniformes (nodos) en un plano acotado de 100x100 y se ejecutan 300 experimentos (10 valores de n, 3 algoritmos, 10 repeticiones), que son 300 rutas en total. Posteriormente, se analiza la distancia total del recorrido y el tiempo de cómputo mediante regresión log-log, intervalos de confianza, gaps de optimalidad y complejidad temporal empírica.

**Resultado principal:** Se demostró que la distancia escala como D(n) ∝ n^0.4 para n = 5..50. La teoría dice que el exponente es igual a 0.5, pero se especula que para que el exponente se aproxime 0.5 el rango de n en el experimento tendría que ser mas grande. Los 3 algoritmos muestran gaps < 12% y tiempos similares.

## Estructura del repositorio
├── resultados/ # Carpeta con el CSV generado y gráficos

├── algoritmos.py # Wrapper de OR-Tools para las 3 estrategias

├── analisis.py # Regresiones, gaps, intervalos de confianza, gráficos

├── experimento.py # Bucle de 300 ejecuciones, guarda CSV

├── generador_puntos.py # Genera puntos aleatorios y matriz euclidiana

├── requirements.txt # Dependencias necesarias

└── README.md # Este archivo

## Tecnologías utilizadas
- Python 3.10+
- OR-Tools (optimización de rutas)
- NumPy, SciPy (regresiones, intervalos de confianza)
- Pandas (manejo de datos)
- Matplotlib (gráficos)

##  Instalación:
pip install -r requirements.txt