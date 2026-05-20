##Análisis Comparativo del Comportamiento Asintótico de Algoritmos para el Problema del Viajante: Escalabilidad, Divergencia y Predicción Mediante Series de Taylor.
# Este módulo genera puntos aleatorios en un plano 100×100

import random
import math

def generar_puntos(n, plano_size=100):
    #n, número de puntos a generar
    #plano_size, tamaño del plano (100×100)
    puntos = []
    
    for _ in range(n):
        x = random.uniform(0, plano_size)
        y = random.uniform(0, plano_size)
        puntos.append((x, y))
    return puntos

def distancia_euclidiana(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def matriz_distancias(puntos):
    n = len(puntos)
    matriz = []
    for i in range(n):
        fila = []
        for j in range(n):
            if i == j:
                fila.append(0)
            else:
                fila.append(distancia_euclidiana(puntos[i], puntos[j]))
        matriz.append(fila)
    return matriz

# ========== CÓDIGO DE PRUEBA ==========
if __name__ == "__main__":
    print("=" * 50)
    print("PRUEBA DEL GENERADOR DE PUNTOS")
    print("=" * 50)
    
    # Generar 10 puntos
    puntos = generar_puntos(10)
    
    print(f"\n📌 Puntos generados (10 puntos):")
    for i, (x, y) in enumerate(puntos):
        print(f"   Punto {i}: ({x:6.2f}, {y:6.2f})")
    
    # Crear matriz de distancias
    matriz = matriz_distancias(puntos)
    
    print(f"\n📌 Matriz de distancias (primeras 5x5):")
    print("      ", end="")
    for j in range(5):
        print(f"  P{j}   ", end="")
    print()
    
    for i in range(5):
        print(f"   P{i} ", end="")
        for j in range(5):
            print(f"{matriz[i][j]:6.2f} ", end="")
        print()
    
    # Estadísticas
    distancias = [matriz[i][j] for i in range(10) for j in range(10) if i != j]
    print(f"\n📊 Estadísticas:")
    print(f"   Distancia mínima: {min(distancias):.2f}")
    print(f"   Distancia máxima: {max(distancias):.2f}")
    print(f"   Distancia promedio: {sum(distancias)/len(distancias):.2f}")
    
    print("\n✅ Generador funcionando correctamente")