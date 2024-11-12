import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Obtención del mapa y conversión de los nodos a una matriz
lugar = "Hohenau, Itapúa, Paraguay"
G = ox.graph_from_place(lugar, network_type="drive")
dist_matrix = nx.floyd_warshall_numpy(G, weight="length")

# Define nodos o rutas que no pueden ser recorridas
nodos_inaccesibles = [5, 12, 20]  # nodos bloqueados
for nodo in nodos_inaccesibles:
    dist_matrix[:, nodo] = np.inf
    dist_matrix[nodo, :] = np.inf

# Asignar un peso de residuos a cada nodo en kg
np.random.seed(42)  # Para reproducibilidad
residuos_por_nodo = {i: np.random.randint(3, 10) for i in range(dist_matrix.shape[0])}
residuos_por_nodo[0] = 0  # Nodo de depósito con residuos 0

# Parámetros del ACO
num_hormigas = 50
num_iteraciones = 30
alpha = 3  # Importancia de la feromona
beta = 2  # Importancia de la visibilidad (1/distancia)
rho = 0.5  # Tasa de evaporación de feromona
Q = 100    # Cantidad de feromona depositada

# Configuración de la recolección
num_puntos = dist_matrix.shape[0]  # Número de puntos de recolección
capacidad_camion = 500  # Capacidad máxima de recolección de un camión (en kg)

# Parámetros de tiempo
hora_inicio = 8  # Hora de inicio de la recolección
hora_limite = 18  # Hora límite para completar la recolección
tiempo_estimado_por_recoleccion = 0.5  # Tiempo estimado por recolección en minutos

# Inicializa feromonas en la red
feromonas = np.ones((num_puntos, num_puntos))

def calcular_probabilidad(punto_actual, puntos_disponibles, dist_matrix):
    """Calcula la probabilidad de seleccionar un punto próximo."""
    global feromonas
    visibilidad = 1 / dist_matrix[punto_actual, puntos_disponibles]
    visibilidad = np.nan_to_num(visibilidad, nan=0.0)  # Evita divisiones por cero
    
    feromonas_actual = feromonas[punto_actual, puntos_disponibles] ** alpha
    visibilidad_actual = visibilidad ** beta
    probabilidades = feromonas_actual * visibilidad_actual
    return probabilidades / np.sum(probabilidades)

def aco_optimizar():
    """Algoritmo principal de ACO para optimización de rutas de recolección."""
    mejor_ruta = None
    mejor_distancia = np.inf
    mejor_distancias_por_generacion = []  # Lista para almacenar las mejores distancias de cada generación
    global feromonas
    
    for _ in range(num_iteraciones):
        rutas = []
        distancias = []
        
        for _ in range(num_hormigas):
            puntos_disponibles = [p for p in range(1, num_puntos) if p not in nodos_inaccesibles]
            ruta = [0]  # Comienza desde el depósito
            distancia_total = 0
            capacidad_actual = capacidad_camion
            hora_actual = hora_inicio 

            while puntos_disponibles and capacidad_actual > 0 and hora_actual < hora_limite:
                punto_actual = ruta[-1]
                probabilidades = calcular_probabilidad(punto_actual, puntos_disponibles, dist_matrix)
                proximo_punto = np.random.choice(puntos_disponibles, p=probabilidades)
                
                # Aumenta el tiempo de recolección
                tiempo_recoleccion = dist_matrix[punto_actual, proximo_punto] / capacidad_camion * tiempo_estimado_por_recoleccion
                hora_actual += tiempo_recoleccion / 60  # Asumimos que cada punto de recolección toma 10 minutos promedio
                
                # Si el tiempo límite es superado, la hormiga no puede continuar
                if hora_actual >= hora_limite:
                    break

                # Actualiza la capacidad del camión y verifica si puede recoger en este punto
                if capacidad_actual >= residuos_por_nodo[proximo_punto]:
                    capacidad_actual -= residuos_por_nodo[proximo_punto]
                    distancia_total += dist_matrix[punto_actual, proximo_punto]
                    ruta.append(proximo_punto)
                    puntos_disponibles.remove(proximo_punto)
                else:
                    break  # Si no hay suficiente capacidad, termina la recolección para esta hormiga

            # Cierra la ruta volviendo al punto de inicio
            if hora_actual < hora_limite:
                distancia_total += dist_matrix[ruta[-1], 0]
                ruta.append(0)
            
            rutas.append(ruta)
            distancias.append(distancia_total)
        
        # Actualiza feromonas
        for ruta, distancia in zip(rutas, distancias):
            for i in range(len(ruta) - 1):
                feromonas[ruta[i], ruta[i+1]] += Q / distancia
                feromonas[ruta[i+1], ruta[i]] += Q / distancia
        
        feromonas *= (1 - rho)  # Evaporación de feromonas
        
        # Actualiza mejor ruta
        indice_mejor_hormiga = np.argmin(distancias)
        if distancias[indice_mejor_hormiga] < mejor_distancia:
            mejor_distancia = distancias[indice_mejor_hormiga]
            mejor_ruta = rutas[indice_mejor_hormiga]
        
        mejor_distancias_por_generacion.append(mejor_distancia)  # Guardamos la mejor distancia por generación

    return mejor_ruta, mejor_distancia, mejor_distancias_por_generacion

# Ejecución del algoritmo
mejor_ruta, mejor_distancia, mejor_distancias_por_generacion = aco_optimizar()
print("Mejor ruta encontrada:", mejor_ruta)
print("Distancia total:", mejor_distancia)

# Graficar la convergencia del algoritmo
plt.plot(range(1, num_iteraciones + 1), mejor_distancias_por_generacion)
plt.xlabel('Generación')
plt.ylabel('Mejor Distancia')
plt.title('Convergencia del Algoritmo ACO')
plt.grid(True)
plt.show()

nodos_osmnx = list(G.nodes)
indice_a_nodo = {i: nodo_id for i, nodo_id in enumerate(nodos_osmnx)}

mejor_ruta_ids = [indice_a_nodo[idx] for idx in mejor_ruta]

nodos_mejor_ruta = [G.nodes[nodo_id] for nodo_id in mejor_ruta_ids]

ruta_latitudes = [nodo['y'] for nodo in nodos_mejor_ruta]
ruta_longitudes = [nodo['x'] for nodo in nodos_mejor_ruta]

fig, ax = ox.plot_graph(G, show=False, close=False)
ax.plot(ruta_longitudes, ruta_latitudes, color='red', linewidth=1, label='Mejor Ruta')
ax.scatter(ruta_longitudes[0], ruta_latitudes[0], color='green', marker='o', s=100, label='Inicio/Fin')  # Punto de inicio/final
ax.legend()
plt.title("Mejor Ruta para Recolección de Residuos")
plt.show()