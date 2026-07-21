# -*- coding: utf-8 -*-
"""
modelo_solar.py
----------------
Implementación DESDE CERO (sin librerías de terceros para el cálculo numérico,
solo `math`) del modelo matemático de geometría solar descrito en el Capítulo I
y el apartado 3.2 de la tesis:

    "Algoritmos genéticos para determinar los ángulos de inclinación óptimos
    de tubos al vacío de termas solares en las principales ciudades de Perú"

El modelo reproduce las ecuaciones (1) a (34) de la tesis, basadas en:
  - Duffie, Beckman & Blair (2020) "Solar Engineering of Thermal Processes,
    Photovoltaics and Wind" (geometría solar, radiación extraterrestre,
    modelo de cielo despejado de Hottel).
  - Tang, Gao y Yu (2009) "Optimal tilt-angles of all-glass evacuated tube
    solar collectors", Energy 34, 1387-1395 (función de aceptación angular,
    factor de forma difuso para colectores tipo T).

La función objetivo (fitness) que usará el Algoritmo Genético es la energía
recolectable ANUAL por unidad de longitud de un tubo (ecuación 34), en MJ/m.
"""

import math

# ---------------------------------------------------------------------------
# Constantes físicas y geométricas (según la tesis)
# ---------------------------------------------------------------------------
GSC = 1367.0          # Constante solar [W/m2]                         (A.7)
D1_MM = 47.0          # Diámetro interior del tubo [mm]  (D1 = 47mm)
D2_MM = 58.0          # Diámetro de cubierta del tubo [mm] (D2 = 58mm)
L_TUBO = 1.80         # Longitud efectiva del tubo al vacío expuesto al sol 1.8 m

# Factores de corrección de Hottel para clima tropical (según la tesis, pág 37)
R0, R1, RK = 0.97, 0.99, 1.02

# Azimut de la superficie: en el hemisferio sur el colector óptimo se orienta
# hacia el norte (hacia el ecuador). Con la convención "0 = sur", orientar
# al norte corresponde a gamma = 180°.
GAMMA_DEG = 180.0


def _rad(deg):
    return math.radians(deg)


def _deg(rad):
    return math.degrees(rad)


# ---------------------------------------------------------------------------
# Geometría del día (ecuaciones 3, 5, 7)
# ---------------------------------------------------------------------------
def dia_angulo_grados(n):
    """Ecuación (3): B = (n-1) * 360/365, en grados."""
    return (n - 1) * 360.0 / 365.0


def radiacion_extraterrestre_Gon(n):
    """Ecuación (5) de Spencer (1971): Gon en W/m2. B se evalúa en grados
    (convención habitual de Duffie & Beckman)."""
    B = _rad(dia_angulo_grados(n))
    return GSC * (1.000110 + 0.034221 * math.cos(B) + 0.001280 * math.sin(B)
                  + 0.00719 * math.cos(2 * B) + 0.000077 * math.sin(2 * B))


def declinacion_grados(n):
    """Ecuación (7) de Spencer (1971), más exacta que la de Cooper (ec. 6).
    B se expresa en RADIANES para este polinomio (forma estándar de Spencer)."""
    B = 2.0 * math.pi * (n - 1) / 365.0
    delta_rad = (0.006918
                 - 0.399912 * math.cos(B)
                 + 0.070257 * math.sin(B)
                 - 0.006758 * math.cos(2 * B)
                 + 0.000907 * math.sin(2 * B)
                 - 0.002697 * math.cos(3 * B)
                 + 0.001480 * math.sin(3 * B))
    return _deg(delta_rad)


# ---------------------------------------------------------------------------
# Ángulos horarios de amanecer/atardecer (ecuaciones 11, 12)
# ---------------------------------------------------------------------------
def _acos_seguro(x):
    return math.acos(max(-1.0, min(1.0, x)))


def angulo_horario_atardecer_horizontal(phi_deg, delta_deg):
    """Ecuación (11): ws (grados) para plano horizontal."""
    phi, delta = _rad(phi_deg), _rad(delta_deg)
    return _deg(_acos_seguro(-math.tan(phi) * math.tan(delta)))


def angulo_horario_atardecer_inclinado(phi_deg, delta_deg, beta_deg):
    """Ecuación (12): wss para un plano inclinado beta.
    Nota de la tesis: para el hemisferio SUR se usa (phi + beta)."""
    ws = angulo_horario_atardecer_horizontal(phi_deg, delta_deg)
    delta = _rad(delta_deg)
    arg = -math.tan(delta) * math.tan(_rad(phi_deg + beta_deg))
    ws_tilt = _deg(_acos_seguro(arg))
    return min(ws, ws_tilt)


# ---------------------------------------------------------------------------
# Radiación directa y difusa (modelo de cielo despejado de Hottel + Liu-Jordan)
# ecuaciones 10, 17-20
# ---------------------------------------------------------------------------
def cos_theta_z(phi_deg, delta_deg, omega_deg):
    """Ecuación (10): coseno del ángulo cenital."""
    phi, delta, omega = _rad(phi_deg), _rad(delta_deg), _rad(omega_deg)
    return math.sin(delta) * math.sin(phi) + math.cos(delta) * math.cos(phi) * math.cos(omega)


def transmitancia_directa(altitud_km, cosz):
    """Ecuación (17): tau_b, modelo de Hottel para cielo despejado."""
    if cosz <= 1e-6:
        return 0.0
    A = altitud_km
    a0 = 0.4237 - 0.00821 * (6 - A) ** 2
    a1 = 0.5055 + 0.00595 * (6.5 - A) ** 2
    k = 0.2711 + 0.01858 * (2.5 - A) ** 2
    return R0 * a0 + R1 * a1 * math.exp(-RK * k / cosz)


def radiacion_directa_Gbn(n, phi_deg, delta_deg, omega_deg, altitud_km):
    """Ecuación (18): Gbn = Gon * cos(thetaZ) * tau_b  [W/m2]."""
    cosz = cos_theta_z(phi_deg, delta_deg, omega_deg)
    if cosz <= 0:
        return 0.0, 0.0
    Gon = radiacion_extraterrestre_Gon(n)
    tau_b = transmitancia_directa(altitud_km, cosz)
    Gbn = Gon * cosz * tau_b
    return Gbn, cosz


def radiacion_difusa_Gd(n, cosz):
    """Ecuaciones (19)-(20): tau_d (Liu & Jordan, 1960) y Gd = Gon*cosz*tau_d."""
    Gon = radiacion_extraterrestre_Gon(n)
    # tau_b se recalcula aquí solo para obtener tau_d (Gbn ya se calculó aparte)
    return Gon, cosz


# ---------------------------------------------------------------------------
# Transformación de coordenadas y ángulos críticos del tubo (ecs. 21-28)
# ---------------------------------------------------------------------------
def vector_solar_ns(phi_deg, delta_deg, omega_deg):
    """Ecuación (21): vector unitario Tierra-Sol (nx, ny, nz)."""
    phi, delta, omega = _rad(phi_deg), _rad(delta_deg), _rad(omega_deg)
    nx = math.cos(delta) * math.cos(phi) * math.cos(omega) + math.sin(delta) * math.sin(phi)
    ny = -math.cos(delta) * math.sin(omega)
    nz = -math.cos(delta) * math.sin(phi) * math.cos(omega) + math.sin(delta) * math.cos(phi)
    return nx, ny, nz


def vector_solar_transformado(nx, ny, nz, beta_deg, gamma_deg=GAMMA_DEG):
    """Ecuación (22): vector unitario Tierra-Sol en el sistema de coordenadas
    del colector (rotado por beta e inclinado según gamma)."""
    beta, gamma = _rad(beta_deg), _rad(gamma_deg)
    nx_p = nx * math.cos(beta) - (ny * math.sin(gamma) + nz * math.cos(gamma)) * math.sin(beta)
    ny_p = ny * math.cos(gamma) - nz * math.sin(gamma)
    nz_p = nx * math.sin(beta) + (ny * math.sin(gamma) + nz * math.cos(gamma)) * math.cos(beta)
    return nx_p, ny_p, nz_p


def cos_theta_t(nx_p, ny_p):
    """Ecuación (24): coseno del ángulo theta_t (colectores tipo T)."""
    val = nx_p * nx_p + ny_p * ny_p
    return math.sqrt(max(val, 0.0))


def angulo_omega_tubo(nx_p, ny_p):
    """Ecuación (25): tan(Omega) = |ny' / nx'|."""
    if abs(nx_p) < 1e-9:
        return math.pi / 2
    return math.atan(abs(ny_p / nx_p))


def angulos_criticos(D1_mm, D2_mm, B_mm):
    """Ecuaciones (27)-(28): Omega0 y Omega1 (en radianes) según distancia B
    entre centros de tubos adyacentes."""
    cos_o0 = (D1_mm + D2_mm) / (2.0 * B_mm)
    cos_o1 = (D1_mm - D2_mm) / (2.0 * B_mm)
    omega0 = _acos_seguro(cos_o0)
    omega1 = _acos_seguro(cos_o1)
    return omega0, omega1


def funcion_aceptacion_f(omega, omega0, omega1, D1_mm, D2_mm, B_mm):
    """Ecuación (26): función de aceptación angular f(Omega)."""
    if omega <= omega0:
        return 1.0
    elif omega <= omega1:
        return (B_mm / D1_mm) * math.cos(omega) + 0.5 * (1 - D2_mm / D1_mm)
    else:
        return 0.0


def factor_forma_F(D1_mm, D2_mm, B_mm):
    """Ecuación (31): factor de forma F para la potencia difusa (colectores tipo T)."""
    omega0, omega1 = angulos_criticos(D1_mm, D2_mm, B_mm)
    F = (omega0 + 0.5 * (1 - D2_mm / D1_mm) * (omega1 - omega0)
         + (B_mm / D1_mm) * (math.sin(omega1) - math.sin(omega0))) / math.pi
    return max(F, 0.0)


# ---------------------------------------------------------------------------
# Potencia total y energía anual (ecuaciones 23, 29, 32, 33, 34)
# ---------------------------------------------------------------------------
def potencia_total_W(n, phi_deg, delta_deg, omega_deg, altitud_km, beta_deg, B_mm,
                      D1_mm=D1_MM, D2_mm=D2_MM, L=L_TUBO):
    """Ecuación (32): Ptotal = D1*Ltubo*(Gbn*cos(theta_t)*f(Omega) + pi*Gdb*F)."""
    Gbn, cosz = radiacion_directa_Gbn(n, phi_deg, delta_deg, omega_deg, altitud_km)
    if cosz <= 0:
        return 0.0

    Gon = radiacion_extraterrestre_Gon(n)
    tau_b = transmitancia_directa(altitud_km, cosz)
    tau_d = 0.271 - 0.294 * tau_b                       # ecuación (19)
    Gd = Gon * cosz * tau_d                             # ecuación (20)
    Gd = max(Gd, 0.0)

    nx, ny, nz = vector_solar_ns(phi_deg, delta_deg, omega_deg)
    nx_p, ny_p, nz_p = vector_solar_transformado(nx, ny, nz, beta_deg)

    if nx_p <= 0:
        # No hay incidencia directa sobre el tubo (ecuación 26, nota f(Omega)=0 si nx'<=0)
        Pbt = 0.0
    else:
        cos_tt = cos_theta_t(nx_p, ny_p)
        omega0, omega1 = angulos_criticos(D1_mm, D2_mm, B_mm)
        omega_t = angulo_omega_tubo(nx_p, ny_p)
        f_omega = funcion_aceptacion_f(omega_t, omega0, omega1, D1_mm, D2_mm, B_mm)
        D1_m = D1_mm / 1000.0
        Pbt = D1_m * L * Gbn * cos_tt * f_omega          # ecuación (23), D1 en metros

    # Potencia difusa (ecuación 29): Pd = D1*Ltubo*pi*Gdb*F
    Gd_beta = 0.5 * (1 + math.cos(_rad(beta_deg))) * Gd   # Gd_beta definido en el texto (sección 1.12 c)
    F = factor_forma_F(D1_mm, D2_mm, B_mm)
    D1_m = D1_mm / 1000.0
    Pd = D1_m * L * math.pi * Gd_beta * F                 # ecuación (29)/(32)

    return Pbt + Pd


def energia_diaria_MJ(n, phi_deg, altitud_km, beta_deg, B_mm, n_pasos=241):
    """Ecuación (33): integra Ptotal(t) entre el amanecer y el atardecer del
    plano inclinado, usando la regla de Simpson (n_pasos impar).
    Devuelve la energía diaria por unidad de longitud, en MJ/m."""
    delta_deg = declinacion_grados(n)
    wss = angulo_horario_atardecer_inclinado(phi_deg, delta_deg, beta_deg)
    if wss <= 0:
        return 0.0

    if n_pasos % 2 == 0:
        n_pasos += 1
    omegas = [-wss + 2 * wss * i / (n_pasos - 1) for i in range(n_pasos)]
    valores = [potencia_total_W(n, phi_deg, delta_deg, w, altitud_km, beta_deg, B_mm) for w in omegas]

    h = (2 * wss) / (n_pasos - 1)     # paso en grados de angulo horario
    # Regla de Simpson compuesta
    suma = valores[0] + valores[-1]
    for i in range(1, n_pasos - 1):
        suma += valores[i] * (4 if i % 2 == 1 else 2)
    integral_grados = suma * h / 3.0   # unidades: W * grado

    # 1 grado de angulo horario = 4 minutos = 240 segundos
    energia_J = integral_grados * 240.0
    return energia_J / 1.0e6           # a MJ/m


def energia_anual_MJ(phi_deg, altitud_km, beta_deg, B_mm=80.0, paso_dias=1, n_pasos=241):
    """Ecuación (34): Hanual = suma_n(Hday), para los 365 días del año.
    paso_dias > 1 y/o n_pasos bajo permiten acelerar el cálculo (usado
    internamente por el AG para evaluar miles de individuos rápidamente);
    la evaluación final de precisión usa paso_dias=1, n_pasos=241."""
    total = 0.0
    dias = list(range(1, 366, paso_dias))
    for n in dias:
        total += energia_diaria_MJ(n, phi_deg, altitud_km, beta_deg, B_mm, n_pasos=n_pasos)
    if paso_dias > 1:
        total *= 365.0 / len(dias)
    return total


# ---------------------------------------------------------------------------
# Datos geográficos de las 6 ciudades (Tabla 2 de la tesis)
# ---------------------------------------------------------------------------
CIUDADES = {
    "TRUJILLO":  {"lat": -8.1117766,  "lon": -79.0286943, "alt_km": 0.034},
    "TACNA":     {"lat": -18.0137712, "lon": -70.2510854, "alt_km": 0.552},
    "AREQUIPA":  {"lat": -16.41545565, "lon": -71.52186355, "alt_km": 2.335},
    "PIURA":     {"lat": -5.197195,   "lon": -80.6266489, "alt_km": 0.055},
    "PUNO":      {"lat": -15.8405816, "lon": -70.0279072, "alt_km": 3.827},
    "IQUITOS":   {"lat": -3.749315,   "lon": -73.2444394, "alt_km": 0.106},
}
