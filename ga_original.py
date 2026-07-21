# -*- coding: utf-8 -*-
"""
ga_original.py
--------------
Implementación DESDE CERO (sin DEAP ni ninguna librería de IA/optimización)
del Algoritmo Genético descrito en la sección 3.4.2 de la tesis.

Réplica de la configuración reportada por el autor (quien usó la librería
DEAP, Distributed Evolutionary Algorithms in Python):

  - Representación   : individuo = un único gen real, el ángulo beta (°)
  - Función objetivo : energía solar recolectable anual (ecuación 34),
                        a maximizar
  - Tamaño población : 50
  - N° generaciones  : 15
  - Cruzamiento      : blend crossover (equivalente a tools.cxBlend de DEAP),
                        prob. 0.7
  - Mutación         : gaussiana (equivalente a tools.mutGaussian de DEAP),
                        prob. 0.1
  - Selección        : torneo (equivalente a tools.selTournament de DEAP)

No se usa NINGUNA librería de terceros: solo el módulo estándar `random`
para los operadores estocásticos y `math`/`time` auxiliares.
"""

import random


class ResultadoGA:
    def __init__(self):
        self.historial = []          # lista de dicts por generación
        self.mejor_individuo = None
        self.mejor_fitness = None


def crear_individuo(rng, lo, hi):
    return [rng.uniform(lo, hi)]


def crear_poblacion(rng, n, lo, hi):
    return [crear_individuo(rng, lo, hi) for _ in range(n)]


def seleccion_torneo(rng, poblacion, fitnesses, k_torneo=3):
    """Selección por torneo (equivalente a tools.selTournament de DEAP)."""
    seleccionados = []
    n = len(poblacion)
    for _ in range(n):
        participantes = [rng.randrange(n) for _ in range(k_torneo)]
        ganador = max(participantes, key=lambda i: fitnesses[i])
        seleccionados.append(list(poblacion[ganador]))
    return seleccionados


def cruzamiento_blend(rng, padre1, padre2, alpha=0.5):
    """Cruzamiento blend (equivalente a tools.cxBlend de DEAP)."""
    x1, x2 = padre1[0], padre2[0]
    gamma = (1.0 + 2 * alpha) * rng.random() - alpha
    hijo1 = x1 + gamma * (x2 - x1)
    gamma = (1.0 + 2 * alpha) * rng.random() - alpha
    hijo2 = x1 + gamma * (x2 - x1)
    return [hijo1], [hijo2]


def mutacion_gaussiana(rng, individuo, sigma, lo, hi):
    """Mutación gaussiana (equivalente a tools.mutGaussian de DEAP)."""
    individuo[0] += rng.gauss(0, sigma)
    individuo[0] = min(max(individuo[0], lo), hi)
    return individuo


def ejecutar_ga(fitness_fn, lo, hi, tam_poblacion=50, n_generaciones=15,
                 prob_cruce=0.7, prob_mutacion=0.1, sigma_mutacion=None,
                 semilla=42, log_callback=None):
    """
    Ejecuta el algoritmo genético original y devuelve un ResultadoGA con el
    historial generación por generación (necesario para el archivo .txt de
    salida solicitado en el trabajo final).

    fitness_fn: función beta -> energía anual (MJ/m) [a maximizar]
    lo, hi    : límites del ángulo de inclinación beta (grados)
    """
    rng = random.Random(semilla)
    if sigma_mutacion is None:
        sigma_mutacion = (hi - lo) * 0.10   # 10% del rango, valor típico DEAP

    resultado = ResultadoGA()

    poblacion = crear_poblacion(rng, tam_poblacion, lo, hi)
    fitnesses = [fitness_fn(ind[0]) for ind in poblacion]

    mejor_global = max(range(tam_poblacion), key=lambda i: fitnesses[i])
    resultado.mejor_individuo = poblacion[mejor_global][0]
    resultado.mejor_fitness = fitnesses[mejor_global]

    def registrar_generacion(gen, poblacion, fitnesses):
        mejor_i = max(range(len(poblacion)), key=lambda i: fitnesses[i])
        peor_i = min(range(len(poblacion)), key=lambda i: fitnesses[i])
        media = sum(fitnesses) / len(fitnesses)
        registro = {
            "generacion": gen,
            "mejor_beta": poblacion[mejor_i][0],
            "mejor_fitness": fitnesses[mejor_i],
            "peor_fitness": fitnesses[peor_i],
            "fitness_promedio": media,
        }
        resultado.historial.append(registro)
        if log_callback:
            log_callback(registro)
        if fitnesses[mejor_i] > resultado.mejor_fitness:
            resultado.mejor_fitness = fitnesses[mejor_i]
            resultado.mejor_individuo = poblacion[mejor_i][0]

    # Generación 0 (población inicial, antes de evolucionar)
    registrar_generacion(0, poblacion, fitnesses)

    for gen in range(1, n_generaciones + 1):
        # 1) Selección
        seleccionados = seleccion_torneo(rng, poblacion, fitnesses)

        # 2) Cruzamiento
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

        # 3) Mutación
        for ind in descendencia:
            if rng.random() < prob_mutacion:
                mutacion_gaussiana(rng, ind, sigma_mutacion, lo, hi)

        # Recorte de límites (por si el cruzamiento se sale del rango)
        for ind in descendencia:
            ind[0] = min(max(ind[0], lo), hi)

        poblacion = descendencia
        fitnesses = [fitness_fn(ind[0]) for ind in poblacion]

        registrar_generacion(gen, poblacion, fitnesses)

    return resultado
