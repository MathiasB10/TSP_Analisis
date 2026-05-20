#Ejecuta 3 algoritmos de OR-Tools y devuelve resultados

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from generador_puntos import matriz_distancias

def ejecutar_algoritmo(matriz_distancias, estrategia):
    #Estrategia se refiere al tipo de busqueda de OR-Tools.
    import time
    n = len(matriz_distancias)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    #RoutingIndexManager configura los parámetros del TSP en órden (número de puntos, número de vehículos, nodo inicial)
    routing = pywrapcp.RoutingModel(manager)
    #RoutingModel es el modelo de rutas que va a resolver el problema, que se crea a partir del manager.
    
    def obtener_distancia(origen, destino):
        origen_original = manager.IndexToNode(origen)
        destino_original = manager.IndexToNode(destino)
        #IndexToNode convierte los índices internos del modelo a los índices originales de matriz_distancias.
        #Multiplicamos por 100 para convertir a enteros, ya que OR-Tools requiere costos enteros.
        return int(matriz_distancias[origen_original][destino_original]*100)

    transit_callback = routing.RegisterTransitCallback(obtener_distancia)
    #RegisterTransitCallback obtiene una función con los índices de OR-Tools transformados a los de matriz_distancias.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)
    #SetArcCostEvaluatorOfAllVehicles asigna transit_callback como el costo de ruta, en este caso distancia.
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = estrategia
    
    start_time = time.time()
    solucion = routing.SolveWithParameters(search_parameters)
    end_time = time.time()
    tiempo = end_time - start_time
    
    if not solucion:
        return None, tiempo, None
    
    indices_ruta = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        indices_ruta.append(manager.IndexToNode(index))
        index = solucion.Value(routing.NextVar(index))
    indices_ruta.append(manager.IndexToNode(index))
    
    distancia_total = 0
    for i in range(len(indices_ruta)-1):
        distancia_total += matriz_distancias[indices_ruta[i]][indices_ruta[i+1]]
    
    return distancia_total, tiempo, indices_ruta