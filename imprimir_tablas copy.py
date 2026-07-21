import json
import modelo_solar as ms

CIUDADES = {
    "TRUJILLO": {"lat": -8.111777, "alt_km": 0.034},
    "PIURA": {"lat": -5.197195, "alt_km": 0.055},
    "IQUITOS": {"lat": -3.749315, "alt_km": 0.106},
    "TACNA": {"lat": -18.013771, "alt_km": 0.552},
    "AREQUIPA": {"lat": -16.415456, "alt_km": 2.335},
    "PUNO": {"lat": -15.840582, "alt_km": 3.827},
}

distancias_B = [80.0, 90.0, 100.0, 110.0, 120.0]
angulos_beta = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

def generar_tablas():
    print("=====================================================================================================")
    print("Resultados de la energía anual con diferentes ángulos y distancias (Como en Tabla 5 y 6)")
    print("=====================================================================================================")
    
    for ciudad, datos in CIUDADES.items():
        print(f"\nUBICACIÓN: {ciudad}")
        cabecera_angulos = "".join([f"b={int(b)}°".ljust(9) for b in angulos_beta])
        print(f"{'B':<8} {cabecera_angulos}")
        print("-" * 80)
        for B in distancias_B:
            fila = f"{int(B)}mm    "
            for beta in angulos_beta:
                energia = ms.energia_anual_MJ(
                    phi_deg=datos["lat"],
                    altitud_km=datos["alt_km"],
                    beta_deg=beta,
                    B_mm=B,
                    paso_dias=1,
                    n_pasos=241
                )
                fila += f"{energia:7.2f}  "
            print(fila)
        print("-" * 80)

if __name__ == "__main__":
    generar_tablas()
