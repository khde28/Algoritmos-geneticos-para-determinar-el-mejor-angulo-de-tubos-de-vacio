# -*- coding: utf-8 -*-
"""
ga_mejorado.py
--------------
MEJORA PROPUESTA al algoritmo genético original de la tesis (sección 3.4.2).

Justificación de la mejora
===========================
La propia tesis (apartado 3.4.4 "Evaluación de Resultados") valida sus
resultados de AG comparándolos contra un método numérico clásico: la
búsqueda de la Sección Áurea (Golden Section Search), un método de
optimización unidimensional que converge de forma muy precisa y rápida
cuando la función es unimodal en el intervalo de búsqueda (como es el caso
de la energía anual en función de beta, según se observa en las Tablas 5 y
6 de la tesis: la energía crece y luego decrece con un único máximo).

Esto sugiere una mejora natural y bien fundamentada: convertir el AG en un
**Algoritmo Memético** (AG + búsqueda local), combinando:

  1. ELITISMO: se conserva siempre el mejor individuo de la generación
     anterior (el AG original de la tesis, al usar reemplazo generacional
     puro estilo DEAP `varAnd` + selección por torneo, PUEDE perder al mejor
     individuo de una generación a la siguiente si no sobrevive el
     cruce/mutación). El elitismo garantiza una convergencia monótona
     (el mejor fitness nunca empeora).

  2. MUTACIÓN GAUSSIANA ADAPTATIVA: sigma decrece linealmente con las
     generaciones (de exploración amplia a explotación fina), en lugar de
     una sigma fija durante las 15 generaciones.

  3. REFINAMIENTO MEMÉTICO (Lamarckiano) mediante BÚSQUEDA DE LA SECCIÓN
     ÁUREA: cada cierto número de generaciones, se aplica una búsqueda de
     la sección áurea EN UN ENTORNO PEQUEÑO alrededor del mejor individuo
     de la población, y el resultado refinado reemplaza al individuo en la
     población (aprendizaje Lamarckiano). Esto acelera enormemente la
     convergencia hacia el óptimo continuo, aprovechando exactamente el
     mismo método que el autor ya usó para VALIDAR sus resultados,
     ahora integrado dentro del propio algoritmo evolutivo.

Esta combinación (AG global + búsqueda local exacta) es un enfoque estándar
y ampliamente documentado en la literatura de optimización evolutiva
(Algoritmos Meméticos, Moscato 1989) para problemas de optimización continua
de baja dimensionalidad como este (una sola variable de decisión: beta).
"""

import random


class ResultadoGA:
    def __init__(self):
        self.historial = []
        self.mejor_individuo = None
        self.mejor_fitness = None


def busqueda_seccion_aurea(fitness_fn, lo, hi, tol=1e-4, max_iter=60):
    """Búsqueda de la Sección Áurea para MAXIMIZAR fitness_fn en [lo, hi].
    (El mismo método que el autor de la tesis usa en la sección 3.4.4 para
    verificar los resultados del AG; aquí se reutiliza como operador de
    búsqueda local dentro del algoritmo memético)."""
    gr = (5 ** 0.5 - 1) / 2.0   # razón áurea inversa ~0.618
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = fitness_fn(c), fitness_fn(d)
    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fitness_fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fitness_fn(d)
        it += 1
    x_opt = (a + b) / 2.0
    return x_opt, fitness_fn(x_opt)


def crear_individuo(rng, lo, hi):
    return [rng.uniform(lo, hi)]


def crear_poblacion(rng, n, lo, hi):
    return [crear_individuo(rng, lo, hi) for _ in range(n)]


def seleccion_torneo(rng, poblacion, fitnesses, k_torneo=3):
    seleccionados = []
    n = len(poblacion)
    for _ in range(n):
        participantes = [rng.randrange(n) for _ in range(k_torneo)]
        ganador = max(participantes, key=lambda i: fitnesses[i])
        seleccionados.append(list(poblacion[ganador]))
    return seleccionados


def cruzamiento_blend(rng, padre1, padre2, alpha=0.5):
    x1, x2 = padre1[0], padre2[0]
    gamma = (1.0 + 2 * alpha) * rng.random() - alpha
    hijo1 = x1 + gamma * (x2 - x1)
    gamma = (1.0 + 2 * alpha) * rng.random() - alpha
    hijo2 = x1 + gamma * (x2 - x1)
    return [hijo1], [hijo2]


def mutacion_gaussiana(rng, individuo, sigma, lo, hi):
    individuo[0] += rng.gauss(0, sigma)
    individuo[0] = min(max(individuo[0], lo), hi)
    return individuo


def ejecutar_ga_mejorado(fitness_fn, lo, hi, tam_poblacion=50, n_generaciones=15,
                          prob_cruce=0.7, prob_mutacion=0.1, sigma_inicial=None,
                          sigma_final_frac=0.15, elitismo=2,
                          periodo_refinamiento=5, radio_refinamiento_frac=0.08,
                          semilla=42, log_callback=None):
    """
    Ejecuta el algoritmo genético MEJORADO (memético + elitista + mutación
    adaptativa) y devuelve el historial generación por generación.
    """
    rng = random.Random(semilla)
    rango = hi - lo
    if sigma_inicial is None:
        sigma_inicial = rango * 0.10
    sigma_final = sigma_inicial * sigma_final_frac

    resultado = ResultadoGA()

    poblacion = crear_poblacion(rng, tam_poblacion, lo, hi)
    fitnesses = [fitness_fn(ind[0]) for ind in poblacion]

    def registrar_generacion(gen, poblacion, fitnesses, sigma_actual, refinado=False):
        mejor_i = max(range(len(poblacion)), key=lambda i: fitnesses[i])
        peor_i = min(range(len(poblacion)), key=lambda i: fitnesses[i])
        media = sum(fitnesses) / len(fitnesses)
        registro = {
            "generacion": gen,
            "mejor_beta": poblacion[mejor_i][0],
            "mejor_fitness": fitnesses[mejor_i],
            "peor_fitness": fitnesses[peor_i],
            "fitness_promedio": media,
            "sigma": sigma_actual,
            "refinamiento_aureo": refinado,
        }
        resultado.historial.append(registro)
        if log_callback:
            log_callback(registro)
        if resultado.mejor_fitness is None or fitnesses[mejor_i] > resultado.mejor_fitness:
            resultado.mejor_fitness = fitnesses[mejor_i]
            resultado.mejor_individuo = poblacion[mejor_i][0]

    registrar_generacion(0, poblacion, fitnesses, sigma_inicial)

    for gen in range(1, n_generaciones + 1):
        frac = gen / n_generaciones
        sigma_actual = sigma_inicial + (sigma_final - sigma_inicial) * frac  # decrece linealmente

        # --- Elitismo: se guardan los "elitismo" mejores individuos ---
        indices_ordenados = sorted(range(len(poblacion)), key=lambda i: fitnesses[i], reverse=True)
        elite = [list(poblacion[i]) for i in indices_ordenados[:elitismo]]

        # --- Selección + cruzamiento + mutación (igual que el AG original) ---
        seleccionados = seleccion_torneo(rng, poblacion, fitnesses)
        descendencia = []
        for i in range(0, tam_poblacion - 1, 2):
            p1, p2 = seleccionados[i], seleccionados[i + 1]
            if rng.random() < prob_cruce:
                h1, h2 = cruzamiento_blend(rng, p1, p2)
            else:
                h1, h2 = list(p1), list(p2)
            descendencia.extend([h1, h2])
        if len(descendencia) < tam_poblacion:
            descendencia.append(list(seleccionados[-1]))

        for ind in descendencia:
            if rng.random() < prob_mutacion:
                mutacion_gaussiana(rng, ind, sigma_actual, lo, hi)
            ind[0] = min(max(ind[0], lo), hi)

        # --- Reinserción elitista: reemplaza a los peores por la élite ---
        descendencia_fitness = [fitness_fn(ind[0]) for ind in descendencia]
        peores_idx = sorted(range(len(descendencia)), key=lambda i: descendencia_fitness[i])[:elitismo]
        for slot, ind_elite in zip(peores_idx, elite):
            descendencia[slot] = ind_elite
            descendencia_fitness[slot] = fitness_fn(ind_elite[0])

        poblacion = descendencia
        fitnesses = descendencia_fitness

        # --- Refinamiento memético (búsqueda de la sección áurea) ---
        refinado = False
        if periodo_refinamiento and gen % periodo_refinamiento == 0:
            mejor_i = max(range(len(poblacion)), key=lambda i: fitnesses[i])
            beta_mejor = poblacion[mejor_i][0]
            radio = rango * radio_refinamiento_frac
            lo_local = max(lo, beta_mejor - radio)
            hi_local = min(hi, beta_mejor + radio)
            beta_refinado, fit_refinado = busqueda_seccion_aurea(fitness_fn, lo_local, hi_local)
            if fit_refinado > fitnesses[mejor_i]:
                poblacion[mejor_i] = [beta_refinado]
                fitnesses[mejor_i] = fit_refinado
                refinado = True

        registrar_generacion(gen, poblacion, fitnesses, sigma_actual, refinado)

    return resultado
