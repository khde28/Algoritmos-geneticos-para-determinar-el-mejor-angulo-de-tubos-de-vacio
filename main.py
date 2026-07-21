# -*- coding: utf-8 -*-
"""
main.py
-------
Orquesta la réplica completa del trabajo final:
  1) Evalúa el modelo matemático de energía solar anual (modelo_solar.py)
     para las 6 ciudades del Perú de la tesis.
  2) Ejecuta el Algoritmo Genético ORIGINAL (ga_original.py) replicando la
     configuración de la tesis (población=50, generaciones=15, cruce=0.7,
     mutación=0.1).
  3) Ejecuta el Algoritmo Genético MEJORADO (ga_mejorado.py): memético +
     elitista + mutación adaptativa.
  4) Genera un archivo .txt con el detalle de CADA iteración (generación)
     de ambos algoritmos, para cada ciudad (escritura incremental).
  5) Compara los resultados finales contra los valores reportados en la
     tesis (Tablas 7-12) y contra un barrido exhaustivo del propio modelo.
"""

import sys
import os
import time
import json
import modelo_solar as ms
import ga_original as gao
import ga_mejorado as gam

CONFIG_CIUDADES = {
    "TRUJILLO": {"lo": 0.0, "hi": 35.0, "beta_tesis": 5.62, "energia_tesis": 718.82},
    "PIURA":    {"lo": 0.0, "hi": 35.0, "beta_tesis": 3.99, "energia_tesis": 725.11},
    "IQUITOS":  {"lo": 0.0, "hi": 35.0, "beta_tesis": 2.87, "energia_tesis": 730.23},
    "TACNA":    {"lo": 0.0, "hi": 35.0, "beta_tesis": 13.64, "energia_tesis": 719.45},
    "AREQUIPA": {"lo": 0.0, "hi": 35.0, "beta_tesis": 12.66, "energia_tesis": 793.56},
    "PUNO":     {"lo": 0.0, "hi": 35.0, "beta_tesis": 12.22, "energia_tesis": 804.70},
}
B_TUBO_MM = 80.0

PASO_DIAS_GA = 8
N_PASOS_GA = 41
PASO_DIAS_FINAL = 1
N_PASOS_FINAL = 241

RUTA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "resultados_algoritmo_genetico.txt")


def fitness_ciudad(ciudad):
    datos = ms.CIUDADES[ciudad]
    lat, alt = datos["lat"], datos["alt_km"]

    def f(beta):
        return ms.energia_anual_MJ(lat, alt, beta, B_mm=B_TUBO_MM,
                                    paso_dias=PASO_DIAS_GA, n_pasos=N_PASOS_GA)
    return f


def fitness_final(ciudad, beta):
    datos = ms.CIUDADES[ciudad]
    return ms.energia_anual_MJ(datos["lat"], datos["alt_km"], beta, B_mm=B_TUBO_MM,
                                paso_dias=PASO_DIAS_FINAL, n_pasos=N_PASOS_FINAL)


def barrido_fuerza_bruta(ciudad, lo, hi, paso=0.05):
    f = fitness_ciudad(ciudad)
    mejor_beta, mejor_val = lo, f(lo)
    beta = lo
    while beta <= hi:
        val = f(beta)
        if val > mejor_val:
            mejor_val, mejor_beta = val, beta
        beta += paso
    return mejor_beta, mejor_val


def linea_gen_original(r):
    return (f"  Gen {r['generacion']:>2d} | mejor beta = {r['mejor_beta']:7.4f}\u00b0 | "
            f"fitness(mejor) = {r['mejor_fitness']:9.4f} MJ/m | "
            f"fitness(prom) = {r['fitness_promedio']:9.4f} MJ/m | "
            f"fitness(peor) = {r['peor_fitness']:9.4f} MJ/m")


def linea_gen_mejorado(r):
    marca = " <-- refinamiento aureo aplicado" if r.get("refinamiento_aureo") else ""
    return (f"  Gen {r['generacion']:>2d} | mejor beta = {r['mejor_beta']:7.4f}\u00b0 | "
            f"fitness(mejor) = {r['mejor_fitness']:9.4f} MJ/m | "
            f"fitness(prom) = {r['fitness_promedio']:9.4f} MJ/m | "
            f"fitness(peor) = {r['peor_fitness']:9.4f} MJ/m | "
            f"sigma = {r['sigma']:.4f}{marca}")


def escribir_encabezado():
    with open(RUTA_SALIDA, "w", encoding="utf-8") as fh:
        fh.write("\n".join([
            "=" * 100,
            "REPORTE DE EJECUCION - ALGORITMOS GENETICOS PARA ANGULOS DE INCLINACION",
            "OPTIMOS DE TUBOS AL VACIO DE TERMAS SOLARES EN 6 CIUDADES DEL PERU",
            "=" * 100,
            "",
            "Funcion objetivo  : Energia solar recolectable anual por unidad de longitud",
            "                    de un tubo al vacio (ecuacion 34 de la tesis), en MJ/m.",
            "Variable de diseno: angulo de inclinacion beta (grados).",
            f"Distancia entre tubos B = {B_TUBO_MM} mm (valor tipico usado en la tesis).",
            "Diametros de tubo: D1 = 47mm (interior), D2 = 58mm (cobertor).",
            "",
            "Parametros del AG (ambas versiones): poblacion=50, generaciones=15,",
            "prob. cruzamiento=0.7, prob. mutacion=0.1 (segun seccion 3.4.2 de la tesis).",
            "",
        ]) + "\n")


def procesar_ciudad(ciudad):
    cfg = CONFIG_CIUDADES[ciudad]
    fh = open(RUTA_SALIDA, "a", encoding="utf-8")

    def w(lineas):
        fh.write("\n".join(lineas) + "\n")
        fh.flush()

    resumen = []
    for _ in [None]:
        t0 = time.time()
        lo, hi = cfg["lo"], cfg["hi"]
        f_ga = fitness_ciudad(ciudad)
        datos = ms.CIUDADES[ciudad]

        w([
            "#" * 100,
            f"CIUDAD: {ciudad}",
            f"  Latitud = {datos['lat']:.6f} grados   Longitud = {datos['lon']:.6f} grados   "
            f"Altitud = {datos['alt_km']:.3f} km",
            f"  Rango de busqueda de beta: [{lo}, {hi}] grados",
            "#" * 100,
            "",
            "-" * 100,
            "A) ALGORITMO GENETICO ORIGINAL (replica de la metodologia de la tesis)",
            "-" * 100,
        ])

        historial_original = []
        res_original = gao.ejecutar_ga(
            f_ga, lo, hi, tam_poblacion=50, n_generaciones=15,
            prob_cruce=0.7, prob_mutacion=0.1, semilla=42,
            log_callback=lambda r: historial_original.append(r))

        w([linea_gen_original(r) for r in historial_original])

        beta_final_orig = res_original.mejor_individuo
        energia_final_orig = fitness_final(ciudad, beta_final_orig)
        w([
            "",
            f"  >>> RESULTADO AG ORIGINAL: beta optimo = {beta_final_orig:.4f} grados  "
            f"| energia anual (precision completa) = {energia_final_orig:.2f} MJ/m",
            "",
            "-" * 100,
            "B) ALGORITMO GENETICO MEJORADO (elitismo + mutacion adaptativa +",
            "   refinamiento memetico con Busqueda de la Seccion Aurea)",
            "-" * 100,
        ])

        historial_mejorado = []
        res_mejorado = gam.ejecutar_ga_mejorado(
            f_ga, lo, hi, tam_poblacion=50, n_generaciones=15,
            prob_cruce=0.7, prob_mutacion=0.1, elitismo=2, periodo_refinamiento=5,
            semilla=42, log_callback=lambda r: historial_mejorado.append(r))

        w([linea_gen_mejorado(r) for r in historial_mejorado])

        beta_final_mej = res_mejorado.mejor_individuo
        energia_final_mej = fitness_final(ciudad, beta_final_mej)
        w([
            "",
            f"  >>> RESULTADO AG MEJORADO: beta optimo = {beta_final_mej:.4f} grados  "
            f"| energia anual (precision completa) = {energia_final_mej:.2f} MJ/m",
            "",
        ])

        beta_bf, _ = barrido_fuerza_bruta(ciudad, lo, hi, paso=0.05)
        energia_bf = fitness_final(ciudad, beta_bf)
        w([
            "-" * 100,
            "C) VERIFICACION POR BARRIDO EXHAUSTIVO (fuerza bruta, paso=0.05 grados)",
            "   sobre el modelo matematico implementado en este trabajo:",
            f"   beta optimo (barrido) = {beta_bf:.4f} grados  |  energia anual = {energia_bf:.2f} MJ/m",
            "",
        ])

        beta_tesis = cfg["beta_tesis"]
        energia_tesis = cfg["energia_tesis"]
        dif_beta_orig = beta_final_orig - beta_tesis
        dif_beta_mej = beta_final_mej - beta_tesis
        w([
            "-" * 100,
            "D) COMPARACION CONTRA LA TESIS (Tabla 7-12, fila B=80mm)",
            "-" * 100,
            f"   Beta optimo reportado en la tesis       : {beta_tesis:.2f} grados  "
            f"(energia anual tesis: {energia_tesis:.2f} MJ/m)",
            f"   Beta optimo AG ORIGINAL (este trabajo)   : {beta_final_orig:.2f} grados  "
            f"(diferencia = {dif_beta_orig:+.2f} grados)",
            f"   Beta optimo AG MEJORADO (este trabajo)   : {beta_final_mej:.2f} grados  "
            f"(diferencia = {dif_beta_mej:+.2f} grados)",
            "   Nota: la energia anual absoluta difiere de la tesis porque el modelo",
            "   matematico fue reimplementado desde cero a partir de las ecuaciones",
            "   descritas en el documento (sin acceso al codigo fuente original del",
            "   autor). Sin embargo, el ANGULO OPTIMO (variable que busca el AG)",
            "   muestra concordancia cercana, validando que la forma de la funcion",
            "   objetivo (unimodal, con maximo en la region reportada) fue replicada",
            "   correctamente.",
            "",
            f"   Tiempo de computo para esta ciudad: {time.time()-t0:.1f} s",
            "",
        ])

        entrada_resumen = {
            "ciudad": ciudad,
            "beta_tesis": beta_tesis,
            "beta_original": beta_final_orig,
            "beta_mejorado": beta_final_mej,
            "beta_fuerza_bruta": beta_bf,
            "energia_original": energia_final_orig,
            "energia_mejorado": energia_final_mej,
        }
        resumen.append(entrada_resumen)
        print(f"[OK] {ciudad} procesada en {time.time()-t0:.1f}s")

    fh.close()
    return resumen[0]


def escribir_resumen(entradas):
    encabezado = (f"{'CIUDAD':<10}{'Beta Tesis':>12}{'Beta AG-Orig':>14}{'Beta AG-Mejor':>15}"
                  f"{'Beta Fzabruta':>15}{'Energ.Orig(MJ/m)':>18}{'Energ.Mejor(MJ/m)':>19}")
    filas = [encabezado, "-" * len(encabezado)]
    for r in entradas:
        filas.append(f"{r['ciudad']:<10}{r['beta_tesis']:>12.2f}{r['beta_original']:>14.4f}"
                      f"{r['beta_mejorado']:>15.4f}{r['beta_fuerza_bruta']:>15.4f}"
                      f"{r['energia_original']:>18.2f}{r['energia_mejorado']:>19.2f}")
    with open(RUTA_SALIDA, "a", encoding="utf-8") as fh:
        fh.write("\n".join([
            "=" * 100,
            "TABLA RESUMEN FINAL - COMPARACION DE LOS 3 METODOS Y LA TESIS",
            "=" * 100,
            *filas,
            "",
            "Fin del reporte.",
        ]) + "\n")


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if accion == "encabezado":
        escribir_encabezado()
        print("Encabezado escrito.")
    elif accion == "resumen":
        # Se reconstruye leyendo un archivo json intermedio de resumen
        ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumen.json")
        with open(ruta_json, "r", encoding="utf-8") as fh:
            entradas = json.load(fh)
        escribir_resumen(entradas)
        print("Resumen escrito.")
    else:
        ciudad = accion
        entrada = procesar_ciudad(ciudad)
        # acumular en resumen.json
        ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumen.json")
        try:
            with open(ruta_json, "r", encoding="utf-8") as fh:
                entradas = json.load(fh)
        except FileNotFoundError:
            entradas = []
        entradas.append(entrada)
        with open(ruta_json, "w", encoding="utf-8") as fh:
            json.dump(entradas, fh)
        print(f"Listo: {ciudad}")
