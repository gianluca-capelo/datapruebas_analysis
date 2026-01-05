import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from neurotask.tmt.preprocessing.interpolation import interpolate_trajectory
from src import config

def parse_neuropruebas_raw(row):
    """Extrae coordenadas crudas (x, y, t) de una fila de Neuropruebas (lógica simplificada del mapper)."""
    try:
        # Extraer tiempos
        raw_times = row.get("cursor_time")
        if isinstance(raw_times, str):
            times = literal_eval(raw_times)
        elif isinstance(raw_times, list):
            times = raw_times
        else:
            return None, None, None
        times = [int(t) for t in times if pd.notna(t)]

        # Extraer posiciones
        raw_positions = row.get("position")
        if isinstance(raw_positions, str):
            positions = literal_eval(raw_positions)
        elif isinstance(raw_positions, list):
            positions = raw_positions
        else:
            return None, None, None
            
        x_list, y_list = [], []
        for p in positions:
            if isinstance(p, str):
                try:
                    x, y = literal_eval(p)
                except:
                    continue
            elif isinstance(p, (list, tuple)):
                x, y = p[0], p[1]
            else:
                continue
            x_list.append(float(x))
            y_list.append(float(y))
            
        # Asegurar misma longitud
        min_len = min(len(x_list), len(times))
        return x_list[:min_len], y_list[:min_len], times[:min_len]
        
    except Exception as e:
        print(f"Error parseando fila: {e}")
        return None, None, None

def calculate_velocity(x, y, t):
    """Calcula velocidad en px/ms."""
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    
    # Diferencias
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    
    # Evitar división por cero en datos crudos
    dt[dt == 0] = 1e-9 
    
    dist = np.sqrt(dx**2 + dy**2)
    velocity = dist / dt
    return t[1:], velocity

def main():
    # 1. Buscar un archivo de sujeto
    search_path = os.path.join(config.DATA_DIR, "raw", "tmt", "neuropruebas", "subjects", "*.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No se encontraron archivos en: {search_path}")
        return

    # Usar el primer archivo encontrado
    file_path = files[10]
    print(f"Analizando archivo: {os.path.basename(file_path)}")
    
    df = pd.read_csv(file_path, on_bad_lines='skip')
    
    # Filtrar solo trials de TMT
    tmt_rows = df[df["trial_type"] == "trail-making-test"]
    
    if tmt_rows.empty:
        print("El archivo no contiene trials 'trail-making-test'")
        return

    # Tomar el último trial (suele ser el más largo/completo)
    row = tmt_rows.iloc[-1]
    
    # 2. Obtener datos CRUDOS
    raw_x, raw_y, raw_t = parse_neuropruebas_raw(row)
    
    if not raw_x or len(raw_x) < 10:
        print("No se pudieron extraer datos suficientes del trial.")
        return

    # 3. INTERPOLAR
    # Nota: Usamos 60Hz 
    interp_x, interp_y, interp_t = interpolate_trajectory(raw_x, raw_y, raw_t, target_freq_hz=60)
    
    print(f"Puntos originales: {len(raw_x)}")
    print(f"Puntos interpolados: {len(interp_x)}")

    # 4. GRAFICAR
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Comparación Interpolación (Sujeto: {os.path.basename(file_path)})", fontsize=16)

    # A. Trayectoria Espacial X-Y
    ax_xy = axes[0, 0]
    ax_xy.plot(raw_x, raw_y, 'o', color='gray', markersize=4, label='Raw (Puntos)')
    ax_xy.plot(interp_x, interp_y, '-', color='red', markersize=4, linewidth=1, label='Interpolado (60Hz)')
    ax_xy.set_title("Trayectoria Espacial (X vs Y)")
    ax_xy.legend()
    ax_xy.invert_yaxis() # Coordenadas de pantalla suelen tener Y invertido

    # B. Posición vs Tiempo (Solo X para no saturar)
    ax_tx = axes[0, 1]
    ax_tx.plot(raw_t, raw_x, '.-', color='gray', alpha=0.5, label='Raw X')
    ax_tx.plot(interp_t, interp_x, '-', color='blue', label='Interp X')
    ax_tx.set_title("Evolución Temporal (X vs Tiempo)")
    ax_tx.set_xlabel("Tiempo (ms)")
    ax_tx.legend()

    # C. Delta Tiempo (Histograma de intervalos)
    ax_dt = axes[1, 0]
    raw_dt = np.diff(raw_t)
    interp_dt = np.diff(interp_t)
    ax_dt.hist(raw_dt, bins=30, alpha=0.5, color='gray', label='Raw dt', density=True)
    ax_dt.axvline(np.mean(interp_dt), color='red', linestyle='--', label=f'Interp dt (~{np.mean(interp_dt):.1f}ms)')
    ax_dt.set_title("Estabilidad del Muestreo (Delta T)")
    ax_dt.set_xlabel("Intervalo entre muestras (ms)")
    ax_dt.legend()

    # D. Perfil de Velocidad
    ax_vel = axes[1, 1]
    t_v_raw, v_raw = calculate_velocity(raw_x, raw_y, raw_t)
    t_v_int, v_int = calculate_velocity(interp_x, interp_y, interp_t)
    
    ax_vel.plot(t_v_raw, v_raw, color='gray', alpha=0.4, label='Velocidad Raw')
    ax_vel.plot(t_v_int, v_int, color='green', linewidth=1.5, label='Velocidad Interp')
    ax_vel.set_title("Perfil de Velocidad")
    ax_vel.set_ylabel("Velocidad (px/ms)")
    ax_vel.set_xlabel("Tiempo (ms)")
    ax_vel.legend()

    plt.tight_layout()
    output_path = os.path.join(config.FIGURES_DIR, "interpolacion_check.png")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nGráfico guardado en: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()