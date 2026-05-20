# Se ejecuta el experimento y se guardan los resultados en un archivo CSV.

import csv
import random
from generador_puntos import generar_puntos, matriz_distancias
from algoritmos import ejecutar_algoritmo
from ortools.constraint_solver import routing_enums_pb2
import os

random.seed(42)
# Se escoge una semilla fija para datos reproducibles.

valores_n = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

algoritmos = [
    ("PATH_CHEAPEST_ARC", routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC),
    ("SAVINGS", routing_enums_pb2.FirstSolutionStrategy.SAVINGS),
    ("CHRISTOFIDES", routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)
]

REPETICIONES = 10
# Se va a ejecutar 10 veces para tener un promedio más confiable.

directorio_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_resultados = os.path.join(directorio_actual, "resultados")
os.makedirs(carpeta_resultados, exist_ok=True)
archivo_salida = os.path.join(carpeta_resultados, "resultados_experimento.csv")
# Se crea la carpeta "resultados" y se manda el CSV ahí.

def ejecutar_experimento():
    with open(archivo_salida, "w", newline="") as csvfile:
        #newline="" evita líneas en blanco entre filas.
        writer = csv.writer(csvfile)
        writer.writerow(["n", "algoritmo", "repetición", "distancia", "tiempo_segundos"])
        
        for n in valores_n:
            print(f"\n📌 Probando n = {n}...")
            
            for nombre_algo, estrategia in algoritmos:
                print(f"Algoritmo: {nombre_algo}")
                
                for rep in range(REPETICIONES):
                    puntos = generar_puntos(n)
                    matriz = matriz_distancias(puntos)
                    distancia, tiempo, _ = ejecutar_algoritmo(matriz, estrategia)
                    writer.writerow([n, nombre_algo, rep+1, distancia, tiempo])
                    if (rep+1) % 5 == 0:
                        print(f"    Progreso: {rep+1}/REPETICIONES")
            print(f"   ✅ Completado para n = {n}")
    
    print(f"\n{'='*50}")
    print(f"✅ Experimento completado!")
    print(f"   Resultados guardados en: {archivo_salida}")
    print(f"   Total de ejecuciones: {len(valores_n) * len(algoritmos) * REPETICIONES}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENTO DE ANÁLISIS DE ALGORITMOS TSP")
    print("=" * 50)
    print(f"Valores de n: {valores_n}")
    print(f"Algoritmos: {[a[0] for a in algoritmos]}")
    print(f"Repeticiones por caso: {REPETICIONES}")
    print(f"Total de ejecuciones: {len(valores_n) * len(algoritmos) * REPETICIONES}")
    print("=" * 50)
    print("\n🚀 Iniciando experimento...\n")

    ejecutar_experimento()