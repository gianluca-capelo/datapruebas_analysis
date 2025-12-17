"""
Script para visualizar los errores de mapeo de neuropruebas y datapruebas.

Genera un gráfico con:
- Distribución de errores por tipo para cada origen
- Proporción de sujetos con errores por origen
- Tabla resumen de estadísticas

Uso:
    python -m src.visualization.plot_mapping_errors
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import LOG_DIR, FIGURES_DIR


def load_error_files():
    """Carga los archivos de errores de mapeo."""
    log_dir = Path(LOG_DIR)
    
    neuro_path = log_dir / "neuropruebas_mapping_errors.csv"
    data_path = log_dir / "datapruebas_mapping_errors.csv"
    
    neuro_errors = None
    data_errors = None
    
    if neuro_path.exists():
        neuro_errors = pd.read_csv(neuro_path)
        print(f"✓ Cargados {len(neuro_errors)} errores de neuropruebas")
    else:
        print(f"⚠ No se encontró {neuro_path}")
    
    if data_path.exists():
        data_errors = pd.read_csv(data_path)
        print(f"✓ Cargados {len(data_errors)} errores de datapruebas")
    else:
        print(f"⚠ No se encontró {data_path}")
    
    return neuro_errors, data_errors


def truncate_error_message(error: str, max_length: int = 50) -> str:
    """Trunca mensajes de error largos."""
    if len(error) > max_length:
        return error[:max_length] + "..."
    return error


def normalize_error_message(error: str) -> str:
    """Normaliza mensajes de error para agruparlos."""
    # Agrupar todos los errores "Error obtaining stimuli for subject ..."
    if error.startswith("Error obtaining stimuli for subject"):
        return "Error obtaining stimuli"
    return error


def plot_mapping_errors(neuro_errors: pd.DataFrame, data_errors: pd.DataFrame, output_path: Path):
    """Genera el gráfico de errores de mapeo."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Errores de Mapeo por Origen', fontsize=16, fontweight='bold')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    
    # 1. Neuropruebas - Error distribution
    ax1 = axes[0, 0]
    if neuro_errors is not None and len(neuro_errors) > 0:
        neuro_error_counts = neuro_errors['error'].apply(
            normalize_error_message
        ).apply(
            lambda x: truncate_error_message(x, 50)
        ).value_counts()
        
        ax1.barh(range(len(neuro_error_counts)), neuro_error_counts.values, 
                 color=colors[:len(neuro_error_counts)])
        ax1.set_yticks(range(len(neuro_error_counts)))
        ax1.set_yticklabels(neuro_error_counts.index, fontsize=8)
        ax1.set_xlabel('Cantidad de sujetos')
        ax1.set_title(f'Neuropruebas\n({len(neuro_errors)} sujetos descartados)', fontsize=12)
        ax1.invert_yaxis()
        
        for i, v in enumerate(neuro_error_counts.values):
            ax1.text(v + 0.5, i, str(v), va='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Sin errores', ha='center', va='center', fontsize=14)
        ax1.set_title('Neuropruebas\n(0 sujetos descartados)', fontsize=12)
    
    # 2. Datapruebas - Error distribution
    ax2 = axes[0, 1]
    if data_errors is not None and len(data_errors) > 0:
        data_error_counts = data_errors['error'].apply(
            lambda x: truncate_error_message(x, 50)
        ).value_counts()
        
        ax2.barh(range(len(data_error_counts)), data_error_counts.values, 
                 color=colors[:len(data_error_counts)])
        ax2.set_yticks(range(len(data_error_counts)))
        ax2.set_yticklabels(data_error_counts.index, fontsize=8)
        ax2.set_xlabel('Cantidad de entradas')
        ax2.set_title(f'Datapruebas\n({len(data_errors)} entradas de error)', fontsize=12)
        ax2.invert_yaxis()
        
        for i, v in enumerate(data_error_counts.values):
            ax2.text(v + 0.1, i, str(v), va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Sin errores', ha='center', va='center', fontsize=14)
        ax2.set_title('Datapruebas\n(0 entradas de error)', fontsize=12)
    
    # 3. Pie chart - Unique subjects with errors
    ax3 = axes[1, 0]
    neuro_unique = neuro_errors['subject_id'].nunique() if neuro_errors is not None else 0
    data_unique = data_errors['subject_id'].nunique() if data_errors is not None else 0
    
    if neuro_unique > 0 or data_unique > 0:
        sizes = [neuro_unique, data_unique]
        labels = [f'Neuropruebas\n({neuro_unique} sujetos)', f'Datapruebas\n({data_unique} sujetos)']
        colors_pie = ['#3498db', '#e74c3c']
        explode = (0.02, 0.02)
        
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 11})
        ax3.set_title('Sujetos únicos con errores\npor origen', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'Sin errores', ha='center', va='center', fontsize=14)
        ax3.set_title('Sujetos únicos con errores', fontsize=12)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    neuro_count = len(neuro_errors) if neuro_errors is not None else 0
    data_count = len(data_errors) if data_errors is not None else 0
    
    neuro_top_count = neuro_errors['error'].value_counts().values[0] if neuro_errors is not None and len(neuro_errors) > 0 else 0
    data_top_count = data_errors['error'].value_counts().values[0] if data_errors is not None and len(data_errors) > 0 else 0
    
    neuro_error_types = neuro_errors['error'].nunique() if neuro_errors is not None and len(neuro_errors) > 0 else 0
    data_error_types = data_errors['error'].nunique() if data_errors is not None and len(data_errors) > 0 else 0
    
    summary_data = [
        ['Total entradas', str(neuro_count), str(data_count)],
        ['Sujetos únicos', str(neuro_unique), str(data_unique)],
        ['Tipos de error', str(neuro_error_types), str(data_error_types)],
        ['Mayor frecuencia', str(neuro_top_count), str(data_top_count)],
    ]
    
    table = ax4.table(
        cellText=summary_data,
        colLabels=['Métrica', 'Neuropruebas', 'Datapruebas'],
        loc='center',
        cellLoc='center',
        colColours=['#ecf0f1'] * 3,
        colWidths=[0.4, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    
    # Ajustar ancho de celdas
    for key, cell in table.get_celld().items():
        cell.set_text_props(wrap=True)
    
    ax4.set_title('Resumen de Errores de Mapeo', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Gráfico guardado en: {output_path}")


def main():
    """Función principal."""
    print("=" * 60)
    print("Generando gráfico de errores de mapeo")
    print("=" * 60)
    
    neuro_errors, data_errors = load_error_files()
    
    if neuro_errors is None and data_errors is None:
        print("❌ No se encontraron archivos de errores. Ejecuta primero el análisis.")
        return
    
    output_path = Path(FIGURES_DIR) / "mapping_errors_distribution.png"
    plot_mapping_errors(neuro_errors, data_errors, output_path)


if __name__ == "__main__":
    main()

