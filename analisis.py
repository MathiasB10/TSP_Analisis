# Análisis de los resultados del experimento TSP
# Este módulo carga el CSV, calcula estadísticas y genera gráficos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os

# Crear carpeta 'resultados/' al inicio, ANTES de cargar datos
carpeta_resultados = "resultados"
if not os.path.exists(carpeta_resultados):
    os.makedirs(carpeta_resultados)
    print(f"📁 Carpeta creada: '{carpeta_resultados}/'")

# Configurar matplotlib para mejor visualización
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ========== CARGAR DATOS ==========

def cargar_datos(archivo="resultados/resultados_experimento.csv"):
    # Carga el CSV y devuelve el DataFrame df con pandas.
    try:
        df = pd.read_csv(archivo, encoding='utf-8')
        print("✅ Archivo leído con UTF-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(archivo, encoding='latin1')
            print("✅ Archivo leído con Latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(archivo, encoding='cp1252')
            print("✅ Archivo leído con CP1252")
    
    print(f"📊 Datos cargados: {len(df)} filas")
    print(f"   Columnas: {list(df.columns)}")
    print(f"   Valores de n: {sorted(df['n'].unique())}")
    print(f"   Algoritmos: {df['algoritmo'].unique()}")
    return df

# ========== CALCULAR PROMEDIOS Y DESVIACIONES ==========

def calcular_promedios(df):
    # Agrupa por n y algoritmo, calcula: promedio de distancia, desviación estándar de distancia, promedio de tiempo, desviación estándar de tiempo, número de repeticiones.
    # Y estos resultados se usan como encabezados de columnas en el nuevo DataFrame promedios.
    promedios = df.groupby(['n', 'algoritmo']).agg(
        distancia_promedio=('distancia', 'mean'),
        distancia_std=('distancia', 'std'),
        tiempo_promedio=('tiempo_segundos', 'mean'),
        tiempo_std=('tiempo_segundos', 'std'),
        repeticiones=('distancia', 'count')
    ).reset_index()
    
    # Calcula coeficiente de variación (estabilidad) y lo añade como nuevo encabezado de columna.
    # Teniendo 8 columnas: n, algoritmo, distancia_promedio, distancia_std, tiempo_promedio, tiempo_std, repeticiones, cv_distancia.
    # Y 30 filas (10 valores de n x 3 algoritmos).
    promedios['cv_distancia'] = promedios['distancia_std'] / promedios['distancia_promedio']
    
    print("\n📈 Promedios calculados:")
    print(promedios.head(10))
    return promedios

# ========== LEY DE POTENCIAS ==========

def ley_de_potencias(promedios):
    print("\n" + "="*60)
    print("📐 LEY DE POTENCIAS: D(n) = a · n^b")
    print("="*60)
    
    resultados = {}
    
    print("📖 Teorema de Beardwood–Halton–Hammersley (1959):")
    print("   Para puntos aleatorios uniformes en el plano,")
    print("   la ruta óptima del TSP crece como β·√(n)")
    print("   → exponente b = 0.5 (asintóticamente)")
    print("   Si b ≈ 0.5, el algoritmo escala como el óptimo teórico.")
    print("   Si b > 0.5, el algoritmo empeora más rápido que lo teórico.\n")

    for algoritmo in promedios['algoritmo'].unique():
        # promedios['algoritmo'].unique() devuelve un array con los nombres de los algoritmos únicos en el DataFrame promedios.
        datos_algo = promedios[promedios['algoritmo'] == algoritmo]
        # promedios['algoritmo'] == algoritmo selecciona las 3 filas donde "algoritmo" es igual al valor actual de la iteración.
        # promedios[...] crea un nuevo DataFrame datos_algo que contiene solo esas 3 filas correspondientes a ese algoritmo.
        n = datos_algo['n'].values
        D = datos_algo['distancia_promedio'].values
        
        # Regresión lineal en log-log
        log_n = np.log(n)
        log_D = np.log(D)
        
        # El teorema BHH dice que 
        # Ajusta el modelo D(n) = a * n^b para cada algoritmo usando regresión lineal sobre log(D) = log(a) + b*log(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_D)
        
        # Pendiente.
        b = slope
        # Intersección (se devuelve el valor original sin log).
        a = np.exp(intercept)
        # Coeficiente de determinación R² para evaluar el ajuste del modelo.
        r2 = r_value**2
        
        resultados[algoritmo] = {
            'a': a,
            'b': b,
            'r2': r2,
            'p_value': p_value,
            'std_err': std_err
        }
        
        # {variable:.nf} para formatear a n decimales, en este caso 2 para a y 3 para b.
        print(f"\n🔹 {algoritmo}:")
        print(f"   D(n) = {a:.2f} · n^{b:.3f}")
        print(f"   R² = {r2:.4f}")
        print(f"   Error estándar de b: {std_err:.4f}")
        
        # Intervalo de confianza del 95% para b con distribución t de Student.
        # Fórmula: media ± t_critico * error estándar de la media. Donde t_critico se obtiene de la distribución t con gl = n-2.
        # Se usa gl = n-2 porque se están estimando 2 parámetros (a y b) en la regresión lineal.
        t_critico = stats.t.ppf(0.975, df=len(n)-2)
        ic_inferior = b - t_critico * std_err
        ic_superior = b + t_critico * std_err
        print(f"   IC 95% para b: [{ic_inferior:.3f}, {ic_superior:.3f}]")
        
        # Verificar si b es significativamente diferente de 0.5 (teorema BHH)
        if ic_inferior <= 0.5 <= ic_superior:
            print(f"   ✅ b = 0.5 dentro del IC → El algoritmo escala como el óptimo teórico (BHH)")
        else:
            if b > 0.5:
                print(f"   ⚠️ b = {b:.3f} > 0.5 → El algoritmo empeora más rápido que el óptimo teórico")
            else:
                print(f"   ⚠️ b = {b:.3f} < 0.5 → El algoritmo mejora más rápido que el óptimo teórico (inusual)")
    
    return resultados

# ========== GAP DE OPTIMALIDAD ==========

def calcular_gaps(promedios):
    # Calcula el gap porcentual entre el mejor y peor algoritmo para cada n
    print("\n" + "="*60)
    print("📊 GAP DE OPTIMALIDAD ENTRE ALGORITMOS")
    print("="*60)
    
    gaps = []
    
    for n in promedios['n'].unique():
        # Valores únicos de n (5, 10, 15, ..., 50)
        datos_n = promedios[promedios['n'] == n]
        # [promedios['n'] == n] selecciona las 3 filas del DataFrame promedios donde n es igual a la iteración actual.
        # Luego, promedios[...] crea un nuevo DataFrame datos_n que contiene solo esas filas.
        
        # datos_n.set_index('algoritmo') hace que cambie el indice de (0, 1, 2...) a (PATH_CHEAPEST_ARC, SAVINGS, CHRISTOFIDES).
        # ["distancia_promedio"] hace una Serie con el índice de algoritmo y los valores de distancia_promedio.
        distancias = datos_n.set_index('algoritmo')['distancia_promedio']
        
        #.indexmin() devuelve el índice del valor mínimo, que corresponde al mejor algoritmo (menor distancia).
        #.idxmax() devuelve el índice del valor máximo, que corresponde al peor algoritmo (mayor distancia).
        mejor_algo = distancias.idxmin()
        peor_algo = distancias.idxmax()
        mejor_dist = distancias.min()
        peor_dist = distancias.max()
        
        gap = (peor_dist - mejor_dist) / mejor_dist * 100
        
        gaps.append({
            'n': n,
            'mejor_algoritmo': mejor_algo,
            'peor_algoritmo': peor_algo,
            'mejor_distancia': mejor_dist,
            'peor_distancia': peor_dist,
            'gap_porcentual': gap
        })
        
        print(f"\n📌 n = {n}:")
        print(f"   Mejor: {mejor_algo} ({mejor_dist:.2f})")
        print(f"   Peor: {peor_algo} ({peor_dist:.2f})")
        print(f"   Gap: {gap:.1f}%")
    
    return pd.DataFrame(gaps)

# ========== ANÁLISIS DE TIEMPOS ==========

def analisis_tiempos(promedios):
    print("\n" + "="*60)
    print("⏱️ ANÁLISIS DE COMPLEJIDAD TEMPORAL")
    print("="*60)
    
    for algoritmo in promedios['algoritmo'].unique():
        datos_algo = promedios[promedios['algoritmo'] == algoritmo]
        n = datos_algo['n'].values
        T = datos_algo['tiempo_promedio'].values
        
        # Regresión log-log para tiempo
        log_n = np.log(n)
        log_T = np.log(T)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_T)
        
        gl = len(n) - 2
        t_critico = stats.t.ppf(0.975, gl)
        ic_inf = slope - t_critico * std_err
        ic_sup = slope + t_critico * std_err
        
        print(f"\n🔹 {algoritmo}:")
        print(f"   T(n) ∝ n^{slope:.2f}")
        print(f"   R² = {r_value**2:.4f}")
        print(f"   Error estándar: {std_err:.4f}")
        print(f"   IC 95% para exponente: [{ic_inf:.2f}, {ic_sup:.2f}]")
        
        # Comparar con complejidad teórica esperada
        # Referencias teóricas (CLRS, bibliografía estándar de algoritmos):
        # - PATH_CHEAPEST_ARC (heurística greedy): O(n²) porque en cada paso busca la arista más barata entre los restantes
        # - SAVINGS (Clarke-Wright): O(n² log n) en implementación eficiente, pero constante alta → aproximadamente n^2.5
        # - CHRISTOFIDES (exacto para métricas): O(n³) por los pasos de árbol mínimo y matching perfecto
        teorico = None
        if algoritmo == 'PATH_CHEAPEST_ARC':
            teorico = 2
            explicacion = "heurística greedy → O(n²)"
        elif algoritmo == 'SAVINGS':
            teorico = 2.5
            explicacion = "Clarke-Wright → O(n² log n) ~ n^2.5 empíricamente"
        elif algoritmo == 'CHRISTOFIDES':
            teorico = 3
            explicacion = "algoritmo de aproximación 3/2 → O(n³)"

        if teorico:
            diff = abs(slope - teorico) / teorico * 100
            print(f"   Teórico esperado: {teorico} ({explicacion})")
            print(f"   Diferencia: {diff:.1f}%")
            if diff < 10:
                print(f"   ✅ Ajuste excelente")
            elif diff < 30:
                print(f"   ⚠️ Desviación moderada (posibles efectos de constante o implementación)")
            else:
                print(f"   ❌ Desviación grande (revisar modelo o datos)")

# ========== CROSSOVER ENTRE ALGORITMOS ==========

def calcular_crossover(promedios, algoritmo_lento='CHRISTOFIDES', algoritmo_rapido='PATH_CHEAPEST_ARC'):
    """
    Encuentra el n donde el algoritmo lento (más preciso) cruza 
    al algoritmo rápido (heurístico) en tiempo de ejecución.
    Responde la pregunta: "¿A partir de qué n conviene usar el rápido?"
    """
    print("\n" + "="*60)
    print("🔄 CROSSOVER DE ALGORITMOS (tiempo de ejecución)")
    print("="*60)
    
    # Obtener datos de cada algoritmo
    datos_lento = promedios[promedios['algoritmo'] == algoritmo_lento]
    datos_rapido = promedios[promedios['algoritmo'] == algoritmo_rapido]
    
    if len(datos_lento) == 0 or len(datos_rapido) == 0:
        print(f"⚠️ No se encontraron datos para {algoritmo_lento} o {algoritmo_rapido}")
        return None
    
    # Regresión log-log para cada uno (tiempo)
    n_lento = datos_lento['n'].values
    T_lento = datos_lento['tiempo_promedio'].values
    log_n_lento = np.log(n_lento)
    log_T_lento = np.log(T_lento)
    slope_lento, intercept_lento, _, _, _ = stats.linregress(log_n_lento, log_T_lento)
    
    n_rapido = datos_rapido['n'].values
    T_rapido = datos_rapido['tiempo_promedio'].values
    log_n_rapido = np.log(n_rapido)
    log_T_rapido = np.log(T_rapido)
    slope_rapido, intercept_rapido, _, _, _ = stats.linregress(log_n_rapido, log_T_rapido)
    
    a_lento = np.exp(intercept_lento)
    a_rapido = np.exp(intercept_rapido)
    b_lento = slope_lento
    b_rapido = slope_rapido
    
    print(f"\n📌 Modelos ajustados:")
    print(f"   {algoritmo_lento}: T(n) = {a_lento:.2e} · n^{b_lento:.2f}")
    print(f"   {algoritmo_rapido}: T(n) = {a_rapido:.2e} · n^{b_rapido:.2f}")
    
    # Resolver a_lento * n^b_lento = a_rapido * n^b_rapido
    # => n = exp( (ln(a_rapido) - ln(a_lento)) / (b_lento - b_rapido) )
    
    if abs(b_lento - b_rapido) < 1e-6:
        print(f"\n⚠️ Los exponentes son casi iguales ({b_lento:.2f} vs {b_rapido:.2f})")
        print("   No hay un crossover claro (las curvas son paralelas en log-log)")
        return None
    
    n_crossover = np.exp((np.log(a_rapido) - np.log(a_lento)) / (b_lento - b_rapido))
    
    print(f"\n🎯 Crossover estimado:")
    print(f"   n ≈ {n_crossover:.1f}")
    print(f"\n📝 Interpretación:")
    if n_crossover < min(min(n_lento), min(n_rapido)):
        print(f"   El crossover ocurre ANTES del n más pequeño medido ({min(min(n_lento), min(n_rapido))})")
        print(f"   → El algoritmo {algoritmo_rapido} es siempre más rápido en el rango medido")
    elif n_crossover > max(max(n_lento), max(n_rapido)):
        print(f"   El crossover ocurre DESPUÉS del n más grande medido ({max(max(n_lento), max(n_rapido))})")
        print(f"   → El algoritmo {algoritmo_lento} es más lento en todo el rango medido")
    else:
        print(f"   • Para n < {n_crossover:.1f}: {algoritmo_rapido} es más rápido")
        print(f"   • Para n > {n_crossover:.1f}: esperarías que {algoritmo_lento} sea más rápido")
        print(f"   (si b_lento > b_rapido, el lento se vuelve más lento; revisa exponentes)")
    
    return n_crossover

# ========== GENERAR GRÁFICOS ==========

def generar_graficos(promedios, gaps_df):
    """
    Genera todos los gráficos del análisis
    """
    print("\n" + "="*60)
    print("📈 GENERANDO GRÁFICOS")
    print("="*60)
    
    # Gráfico 1: Distancia vs n
    plt.figure()
    for algoritmo in promedios['algoritmo'].unique():
        datos = promedios[promedios['algoritmo'] == algoritmo]
        plt.errorbar(datos['n'], datos['distancia_promedio'], 
                    yerr=datos['distancia_std'], 
                    marker='o', capsize=3, label=algoritmo)
    plt.xlabel('Número de puntos (n)')
    plt.ylabel('Distancia promedio')
    plt.title('Comparación de Algoritmos TSP: Distancia vs n')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(carpeta_resultados, 'grafico_distancia_vs_n.png'), dpi=150, bbox_inches='tight')
    print("✅ Gráfico 1 guardado: grafico_distancia_vs_n.png")
    
    # Gráfico 2: Gap porcentual
    plt.figure()
    plt.plot(gaps_df['n'], gaps_df['gap_porcentual'], 'o-', color='red', linewidth=2)
    plt.xlabel('Número de puntos (n)')
    plt.ylabel('Gap porcentual (%)')
    plt.title('Divergencia entre el mejor y peor algoritmo')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(carpeta_resultados, 'grafico_gap_vs_n.png'), dpi=150, bbox_inches='tight')
    print("✅ Gráfico 2 guardado: grafico_gap_vs_n.png")
    
    # Gráfico 3: Log-log para ley de potencias
    plt.figure()
    for algoritmo in promedios['algoritmo'].unique():
        datos = promedios[promedios['algoritmo'] == algoritmo]
        plt.loglog(datos['n'], datos['distancia_promedio'], 'o-', label=algoritmo)
    plt.xlabel('log(n)')
    plt.ylabel('log(Distancia)')
    plt.title('Ley de Potencias: log(D) vs log(n)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(carpeta_resultados, 'grafico_loglog_distancia.png'), dpi=150, bbox_inches='tight')
    print("✅ Gráfico 3 guardado: grafico_loglog_distancia.png")
    
    # Gráfico 4: Tiempo de cómputo
    plt.figure()
    for algoritmo in promedios['algoritmo'].unique():
        datos = promedios[promedios['algoritmo'] == algoritmo]
        plt.loglog(datos['n'], datos['tiempo_promedio'], 'o-', label=algoritmo)
    plt.xlabel('log(n)')
    plt.ylabel('log(Tiempo) [segundos]')
    plt.title('Complejidad Temporal Empírica')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(carpeta_resultados, 'grafico_complejidad_temporal.png'), dpi=150, bbox_inches='tight')
    print("✅ Gráfico 4 guardado: grafico_complejidad_temporal.png")
    
    # Gráfico 5: Coeficiente de variación (estabilidad)
    plt.figure()
    for algoritmo in promedios['algoritmo'].unique():
        datos = promedios[promedios['algoritmo'] == algoritmo]
        plt.plot(datos['n'], datos['cv_distancia'], 'o-', label=algoritmo)
    plt.xlabel('Número de puntos (n)')
    plt.ylabel('Coeficiente de variación (σ/μ)')
    plt.title('Estabilidad de Algoritmos (menor = más estable)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(carpeta_resultados, 'grafico_estabilidad.png'), dpi=150, bbox_inches='tight')
    print("✅ Gráfico 5 guardado: grafico_estabilidad.png")
    
    print("\n📁 Todos los gráficos guardados en la carpeta actual")

# ========== EJECUCIÓN PRINCIPAL ==========

if __name__ == "__main__":
    print("="*60)
    print("ANÁLISIS DE RESULTADOS - TSP EXPERIMENTAL")
    print("="*60)
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Calcular promedios
    promedios = calcular_promedios(df)
    
    # 3. Ley de potencias
    resultados_ley = ley_de_potencias(promedios)
    
    # 4. Gaps
    gaps_df = calcular_gaps(promedios)
    
    # 5. Análisis de tiempos
    analisis_tiempos(promedios)
    
    # 6. Crossover
    crossover_n = calcular_crossover(promedios)
    
    # 7. Generar gráficos
    generar_graficos(promedios, gaps_df)
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)