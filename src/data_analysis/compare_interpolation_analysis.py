#!/usr/bin/env python3
"""
Comparar análisis sin interpolación vs con interpolación.

Compara:
- Análisis 1: Sin interpolación (INTERPOLATE_TRAJECTORY=False)
- Análisis 2: Con interpolación (INTERPOLATE_TRAJECTORY=True)

Uso:
    # Comparación básica (solo texto)
    python -m src.data_analysis.compare_interpolation_analysis

    # Comparación completa con visualizaciones
    python -m src.data_analysis.compare_interpolation_analysis --visualize

    # Comparación custom
    python -m src.data_analysis.compare_interpolation_analysis \
        --analysis1 2026-01-13_09-28-08 \
        --analysis2 2026-01-12_10-13-37 \
        --name1 "Nuevo" \
        --name2 "Viejo" \
        --visualize
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_analysis_data(analysis_path: str) -> pd.DataFrame:
    """Cargar datos de análisis desde CSV."""
    csv_path = Path(analysis_path) / "analysis.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    return pd.read_csv(csv_path)


def compare_structure(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    """Comparar estructura básica de los DataFrames."""
    print("\n" + "="*80)
    print("1. ANÁLISIS BRUTO DE DIFERENCIAS ESTRUCTURALES")
    print("="*80)
    
    print(f"\n{'Métrica':<40} {name1:<25} {name2:<25}")
    print("-"*80)
    print(f"{'Total Trials':<40} {len(df1):<25,} {len(df2):<25,}")
    print(f"{'Sujetos únicos':<40} {df1['subject_id'].nunique():<25} {df2['subject_id'].nunique():<25}")
    print(f"{'Trials/sujeto (promedio)':<40} {len(df1)/df1['subject_id'].nunique():<25.2f} {len(df2)/df2['subject_id'].nunique():<25.2f}")
    
    # Comparar columnas
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"\n{'Columnas':<40} {'Cantidad':<25}")
    print("-"*80)
    print(f"{'Columnas comunes':<40} {len(common_cols):<25}")
    print(f"{'Solo en ' + name1:<40} {len(only_in_1):<25}")
    print(f"{'Solo en ' + name2:<40} {len(only_in_2):<25}")
    
    if only_in_1:
        print(f"\nColumnas solo en {name1}:")
        for col in sorted(only_in_1):
            print(f"  - {col}")
    
    if only_in_2:
        print(f"\nColumnas solo en {name2}:")
        for col in sorted(only_in_2):
            print(f"  - {col}")
    
    # Comparar experiment_origin
    if 'experiment_origin' in common_cols:
        print(f"\nDistribución por experiment_origin:")
        print(f"{'Origen':<30} {name1:<25} {name2:<25}")
        print("-"*80)
        for origin in ['datapruebas', 'neuropruebas']:
            count1 = len(df1[df1['experiment_origin'] == origin]) if origin in df1['experiment_origin'].values else 0
            count2 = len(df2[df2['experiment_origin'] == origin]) if origin in df2['experiment_origin'].values else 0
            print(f"{origin:<30} {count1:<25,} {count2:<25,}")
    
    # Comparar invalid_cause
    if 'invalid_cause' in common_cols:
        print(f"\nDistribución de invalid_cause:")
        invalid1 = df1['invalid_cause'].value_counts(dropna=False).sort_index()
        invalid2 = df2['invalid_cause'].value_counts(dropna=False).sort_index()
        
        all_causes = sorted(set(invalid1.index) | set(invalid2.index), 
                           key=lambda x: (x is None, str(x) if pd.notna(x) else ''))
        
        print(f"{'Tipo':<30} {name1:<25} {name2:<25} {'Diferencia':<15}")
        print("-"*80)
        for cause in all_causes:
            count1 = invalid1.get(cause, 0)
            count2 = invalid2.get(cause, 0)
            diff = count2 - count1
            cause_display = 'Valid (NaN)' if pd.isna(cause) else str(cause)
            print(f"{cause_display:<30} {count1:<25} {count2:<25} {diff:+d}")
    
    return common_cols, only_in_1, only_in_2


def compare_numeric_metrics(df1: pd.DataFrame, df2: pd.DataFrame, common_cols: set, 
                            name1: str, name2: str):
    """Comparar métricas numéricas entre los dos DataFrames."""
    print("\n" + "="*80)
    print("2. COMPARACIÓN DE MÉTRICAS NUMÉRICAS")
    print("="*80)
    
    # Filtrar solo columnas numéricas comunes
    numeric_cols = []
    for col in common_cols:
        if col in ['subject_id', 'trial_id']:
            continue
        if df1[col].dtype in [np.int64, np.float64] or df1[col].dtype == 'object':
            try:
                pd.to_numeric(df1[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
    
    print(f"\nComparando {len(numeric_cols)} columnas numéricas...")
    
    differences = []
    
    for col in sorted(numeric_cols):
        try:
            # Convertir a numérico, ignorando valores no numéricos
            series1 = pd.to_numeric(df1[col], errors='coerce')
            series2 = pd.to_numeric(df2[col], errors='coerce')
            
            # Filtrar NaN
            valid1 = series1.dropna()
            valid2 = series2.dropna()
            
            if len(valid1) == 0 or len(valid2) == 0:
                continue
            
            # Calcular estadísticas
            stats1 = {
                'mean': valid1.mean(),
                'std': valid1.std(),
                'min': valid1.min(),
                'max': valid1.max(),
                'median': valid1.median(),
                'count': len(valid1)
            }
            
            stats2 = {
                'mean': valid2.mean(),
                'std': valid2.std(),
                'min': valid2.min(),
                'max': valid2.max(),
                'median': valid2.median(),
                'count': len(valid2)
            }
            
            # Calcular diferencias relativas (solo si mean != 0)
            mean_diff_pct = ((stats2['mean'] - stats1['mean']) / stats1['mean'] * 100) if stats1['mean'] != 0 else 0
            
            differences.append({
                'column': col,
                'mean_diff': stats2['mean'] - stats1['mean'],
                'mean_diff_pct': mean_diff_pct,
                'mean1': stats1['mean'],
                'mean2': stats2['mean'],
                'std1': stats1['std'],
                'std2': stats2['std'],
                'count1': stats1['count'],
                'count2': stats2['count']
            })
            
        except Exception as e:
            continue
    
    # Ordenar por diferencia absoluta
    differences_df = pd.DataFrame(differences)
    if len(differences_df) > 0:
        differences_df['abs_mean_diff_pct'] = differences_df['mean_diff_pct'].abs()
        differences_df = differences_df.sort_values('abs_mean_diff_pct', ascending=False)
        
        print(f"\nTop 20 columnas con mayores diferencias (%):")
        print(f"{'Columna':<40} {'Mean ' + name1:<20} {'Mean ' + name2:<20} {'Diff %':<15}")
        print("-"*95)
        
        for _, row in differences_df.head(20).iterrows():
            print(f"{row['column']:<40} {row['mean1']:<20.4f} {row['mean2']:<20.4f} {row['mean_diff_pct']:<15.2f}%")
        
        return differences_df
    else:
        print("\nNo se encontraron columnas numéricas para comparar.")
        return pd.DataFrame()


def compare_by_subject(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    """Comparar análisis por sujeto."""
    print("\n" + "="*80)
    print("3. COMPARACIÓN POR SUJETO Y TRIAL")
    print("="*80)
    
    subjects1 = set(df1['subject_id'].unique())
    subjects2 = set(df2['subject_id'].unique())
    
    common_subjects = subjects1 & subjects2
    only_in_1 = subjects1 - subjects2
    only_in_2 = subjects2 - subjects1
    
    print(f"\nSujetos en {name1}: {len(subjects1)}")
    print(f"Sujetos en {name2}: {len(subjects2)}")
    print(f"Sujetos comunes: {len(common_subjects)}")
    print(f"Solo en {name1}: {len(only_in_1)}")
    print(f"Solo en {name2}: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\nSujetos solo en {name1} (primeros 10):")
        for subj in list(only_in_1)[:10]:
            print(f"  - {subj}")
    
    if only_in_2:
        print(f"\nSujetos solo en {name2} (primeros 10):")
        for subj in list(only_in_2)[:10]:
            print(f"  - {subj}")
    
    # Comparar número de trials por sujeto
    if common_subjects:
        print(f"\nComparando número de trials por sujeto (sujetos comunes):")
        trials_per_subj1 = df1[df1['subject_id'].isin(common_subjects)].groupby('subject_id').size()
        trials_per_subj2 = df2[df2['subject_id'].isin(common_subjects)].groupby('subject_id').size()
        
        diff_trials = trials_per_subj2 - trials_per_subj1
        print(f"Sujetos con diferente número de trials: {(diff_trials != 0).sum()}")
        if (diff_trials != 0).any():
            print("\nTop 10 sujetos con mayor diferencia en número de trials:")
            print(f"{'Subject ID':<50} {name1:<15} {name2:<15} {'Diferencia':<15}")
            print("-"*95)
            for subj, diff in diff_trials.abs().nlargest(10).items():
                count1 = trials_per_subj1.get(subj, 0)
                count2 = trials_per_subj2.get(subj, 0)
                print(f"{subj:<50} {count1:<15} {count2:<15} {count2-count1:+d}")
    
    return common_subjects, only_in_1, only_in_2


def deep_analysis_differences(df1: pd.DataFrame, df2: pd.DataFrame, differences_df: pd.DataFrame,
                              common_cols: set, name1: str, name2: str):
    """Análisis profundo de diferencias encontradas."""
    print("\n" + "="*80)
    print("4. ANÁLISIS PROFUNDO DE DIFERENCIAS")
    print("="*80)
    
    if len(differences_df) == 0:
        print("\nNo se encontraron diferencias numéricas significativas para analizar.")
        return
    
    # Filtrar columnas con diferencias significativas (>1% de diferencia)
    significant = differences_df[differences_df['abs_mean_diff_pct'] > 1.0]
    
    if len(significant) == 0:
        print("\nNo se encontraron columnas con diferencias significativas (>1%).")
        return
    
    print(f"\nColumnas con diferencias significativas (>1%): {len(significant)}")
    
    # Analizar cada columna significativa
    for _, row in significant.head(10).iterrows():
        col = row['column']
        print(f"\n{'='*80}")
        print(f"Análisis detallado: {col}")
        print(f"{'='*80}")
        print(f"Diferencia promedio: {row['mean_diff']:.4f} ({row['mean_diff_pct']:.2f}%)")
        
        # Comparar por experiment_origin si existe
        if 'experiment_origin' in common_cols:
            print(f"\nPor experiment_origin:")
            for origin in ['datapruebas', 'neuropruebas']:
                df1_orig = df1[df1['experiment_origin'] == origin][col]
                df2_orig = df2[df2['experiment_origin'] == origin][col]
                
                df1_orig_numeric = pd.to_numeric(df1_orig, errors='coerce').dropna()
                df2_orig_numeric = pd.to_numeric(df2_orig, errors='coerce').dropna()
                
                if len(df1_orig_numeric) > 0 and len(df2_orig_numeric) > 0:
                    mean1 = df1_orig_numeric.mean()
                    mean2 = df2_orig_numeric.mean()
                    diff_pct = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
                    print(f"  {origin:<20}: {name1} mean={mean1:.4f}, {name2} mean={mean2:.4f}, diff={diff_pct:.2f}%")
        
        # Comparar por invalid_cause si existe
        if 'invalid_cause' in common_cols:
            print(f"\nPor invalid_cause (solo válidos):")
            df1_valid = pd.to_numeric(df1[df1['invalid_cause'].isna()][col], errors='coerce').dropna()
            df2_valid = pd.to_numeric(df2[df2['invalid_cause'].isna()][col], errors='coerce').dropna()
            
            if len(df1_valid) > 0 and len(df2_valid) > 0:
                mean1 = df1_valid.mean()
                mean2 = df2_valid.mean()
                diff_pct = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
                print(f"  Valid trials: {name1} mean={mean1:.4f}, {name2} mean={mean2:.4f}, diff={diff_pct:.2f}%")


def generate_report(df1: pd.DataFrame, df2: pd.DataFrame, differences_df: pd.DataFrame,
                    common_subjects: set, name1: str, name2: str, output_path: Path = None):
    """Generar reporte final."""
    print("\n" + "="*80)
    print("5. REPORTE EJECUTIVO")
    print("="*80)
    
    print(f"\nResumen de comparación:")
    print(f"  - Total trials: {name1}={len(df1):,}, {name2}={len(df2):,}, diferencia={len(df2)-len(df1):+,}")
    print(f"  - Sujetos únicos: {name1}={df1['subject_id'].nunique()}, {name2}={df2['subject_id'].nunique()}")
    print(f"  - Sujetos comunes: {len(common_subjects)}")
    
    if len(differences_df) > 0:
        significant = differences_df[differences_df['abs_mean_diff_pct'] > 1.0]
        print(f"\n  - Columnas con diferencias significativas (>1%): {len(significant)}")
        if len(significant) > 0:
            print(f"\n  Top 5 diferencias más grandes:")
            for _, row in significant.head(5).iterrows():
                print(f"    • {row['column']}: {row['mean_diff_pct']:.2f}%")
    
    if output_path:
        print(f"\nReporte guardado en: {output_path}")
        with open(output_path, 'w') as f:
            f.write(f"Comparación de análisis\n")
            f.write(f"{name1} vs {name2}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total trials: {len(df1):,} vs {len(df2):,}\n")
            f.write(f"Sujetos: {df1['subject_id'].nunique()} vs {df2['subject_id'].nunique()}\n")
            if len(differences_df) > 0:
                significant = differences_df[differences_df['abs_mean_diff_pct'] > 1.0]
                f.write(f"\nColumnas con diferencias >1%: {len(significant)}\n")
                for _, row in significant.iterrows():
                    f.write(f"  {row['column']}: {row['mean_diff_pct']:.2f}%\n")


def create_visualizations(df1: pd.DataFrame, df2: pd.DataFrame, differences_df: pd.DataFrame,
                          common_cols: set, name1: str, name2: str, output_dir: Path):
    """
    Generar visualizaciones comprehensivas de las diferencias entre análisis.

    Args:
        df1, df2: DataFrames de ambos análisis
        differences_df: DataFrame con diferencias calculadas
        common_cols: Columnas comunes entre ambos análisis
        name1, name2: Nombres descriptivos de los análisis
        output_dir: Directorio donde guardar las figuras

    Returns:
        List[Path]: Lista de paths a las figuras generadas
    """
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })

    # Crear directorio para figuras
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    # Filtrar diferencias significativas (>1%)
    significant = differences_df[differences_df['abs_mean_diff_pct'] > 1.0].copy()

    if len(significant) == 0:
        print("⚠️  No se encontraron diferencias significativas (>1%) para visualizar.")
        return figure_paths

    # =============================================================================
    # 1. TOP DIFFERENCES BAR PLOT
    # =============================================================================
    print("Generando gráfico de top differences...")
    fig, ax = plt.subplots(figsize=(12, 8))

    top_n = min(20, len(significant))
    top_diff = significant.head(top_n).sort_values('mean_diff_pct', ascending=True)

    colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_diff['mean_diff_pct']]
    ax.barh(range(top_n), top_diff['mean_diff_pct'], color=colors, alpha=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_diff['column'], fontsize=9)
    ax.set_xlabel(f'Diferencia Porcentual (%) - {name2} vs {name1}', fontsize=11)
    ax.set_title(f'Top {top_n} Métricas con Mayor Diferencia', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "01_top_differences.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths.append(fig_path)

    # =============================================================================
    # 2. DISTRIBUTION COMPARISONS (Top 6 métricas)
    # =============================================================================
    print("Generando comparaciones de distribuciones...")
    top_metrics = significant.head(6)['column'].tolist()

    for i, metric in enumerate(top_metrics):
        # Convertir a numérico
        series1 = pd.to_numeric(df1[metric], errors='coerce').dropna()
        series2 = pd.to_numeric(df2[metric], errors='coerce').dropna()

        if len(series1) == 0 or len(series2) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histograma superpuesto
        axes[0].hist(series1, bins=30, alpha=0.6, label=name1, color='blue', density=True)
        axes[0].hist(series2, bins=30, alpha=0.6, label=name2, color='red', density=True)
        axes[0].set_xlabel(metric, fontsize=10)
        axes[0].set_ylabel('Densidad', fontsize=10)
        axes[0].set_title('Distribución Superpuesta', fontsize=11)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot
        data_box = pd.DataFrame({
            name1: series1,
            name2: series2
        })
        data_box_melted = data_box.melt(var_name='Análisis', value_name='Valor')
        sns.boxplot(data=data_box_melted, x='Análisis', y='Valor', ax=axes[1], palette=['blue', 'red'])
        axes[1].set_title('Box Plot Comparativo', fontsize=11)
        axes[1].set_ylabel(metric, fontsize=10)
        axes[1].grid(alpha=0.3)

        # Violin plot
        sns.violinplot(data=data_box_melted, x='Análisis', y='Valor', ax=axes[2], palette=['blue', 'red'])
        axes[2].set_title('Violin Plot Comparativo', fontsize=11)
        axes[2].set_ylabel(metric, fontsize=10)
        axes[2].grid(alpha=0.3)

        fig.suptitle(f'Comparación de Distribuciones: {metric}', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        fig_path = output_dir / f"02_distributions_{metric.replace('/', '_')[:40]}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths.append(fig_path)

    # =============================================================================
    # 3. SCATTER PLOTS (Top 6 métricas)
    # =============================================================================
    print("Generando scatter plots...")

    for i, metric in enumerate(top_metrics):
        if metric not in df1.columns or metric not in df2.columns:
            continue

        # Preparar datos
        data_scatter = pd.DataFrame({
            'value1': pd.to_numeric(df1[metric], errors='coerce'),
            'value2': pd.to_numeric(df2[metric], errors='coerce'),
        })

        # Agregar experiment_origin si está disponible
        if 'experiment_origin' in common_cols:
            data_scatter['origin'] = df1['experiment_origin'].values

        data_scatter = data_scatter.dropna(subset=['value1', 'value2'])

        if len(data_scatter) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        if 'origin' in data_scatter.columns:
            for origin, color in zip(['datapruebas', 'neuropruebas'], ['#1f77b4', '#ff7f0e']):
                mask = data_scatter['origin'] == origin
                if mask.any():
                    ax.scatter(data_scatter.loc[mask, 'value1'],
                               data_scatter.loc[mask, 'value2'],
                               alpha=0.5, s=30, label=origin, color=color)
        else:
            ax.scatter(data_scatter['value1'], data_scatter['value2'],
                       alpha=0.5, s=30, color='#1f77b4')

        # Línea diagonal y=x
        lims = [
            min(data_scatter['value1'].min(), data_scatter['value2'].min()),
            max(data_scatter['value1'].max(), data_scatter['value2'].max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='y=x (sin cambio)')

        ax.set_xlabel(f'{metric} - {name1}', fontsize=11)
        ax.set_ylabel(f'{metric} - {name2}', fontsize=11)
        ax.set_title(f'Scatter Plot: {metric}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / f"03_scatter_{metric.replace('/', '_')[:40]}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths.append(fig_path)

    # =============================================================================
    # 4. CORRELATION HEATMAP OF DIFFERENCES
    # =============================================================================
    print("Generando correlation heatmap...")

    # Calcular diferencias para todas las métricas significativas
    diff_data = {}
    for _, row in significant.iterrows():
        metric = row['column']
        if metric in df1.columns and metric in df2.columns:
            series1 = pd.to_numeric(df1[metric], errors='coerce')
            series2 = pd.to_numeric(df2[metric], errors='coerce')
            diff = series2 - series1
            diff_data[metric] = diff

    if len(diff_data) > 1:
        diff_df = pd.DataFrame(diff_data).dropna()

        # Tomar top 15 para que el heatmap no sea demasiado grande
        top_for_heatmap = min(15, len(diff_df.columns))
        corr_matrix = diff_df.iloc[:, :top_for_heatmap].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    ax=ax)
        ax.set_title(f'Correlación entre Diferencias (Top {top_for_heatmap} métricas)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        fig_path = output_dir / "04_correlation_heatmap.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths.append(fig_path)

    # =============================================================================
    # 5. DIFFERENCE DISTRIBUTIONS (Histogramas de diferencias)
    # =============================================================================
    print("Generando histogramas de diferencias...")

    top_for_diff_hist = min(9, len(significant))
    top_diff_metrics = significant.head(top_for_diff_hist)['column'].tolist()

    rows = (top_for_diff_hist + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten() if top_for_diff_hist > 1 else [axes]

    for idx, metric in enumerate(top_diff_metrics):
        if metric in df1.columns and metric in df2.columns:
            series1 = pd.to_numeric(df1[metric], errors='coerce')
            series2 = pd.to_numeric(df2[metric], errors='coerce')
            diff = (series2 - series1).dropna()

            if len(diff) > 0:
                axes[idx].hist(diff, bins=40, color='purple', alpha=0.7, edgecolor='black')
                axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
                axes[idx].set_xlabel('Diferencia (Análisis 2 - Análisis 1)', fontsize=9)
                axes[idx].set_ylabel('Frecuencia', fontsize=9)
                axes[idx].set_title(f'{metric}\n(Mean diff: {diff.mean():.3f})', fontsize=10)
                axes[idx].grid(alpha=0.3)

    # Ocultar ejes no usados
    for idx in range(top_for_diff_hist, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Distribución de Diferencias por Métrica', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    fig_path = output_dir / "05_difference_histograms.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths.append(fig_path)

    return figure_paths


def main():
    """Función principal."""
    from src.config import DATA_DIR

    # Parsear argumentos CLI
    parser = argparse.ArgumentParser(
        description="Comparar dos análisis TMT para evaluar diferencias"
    )
    parser.add_argument(
        '--analysis1',
        default='2026-01-13_09-28-08',
        help='Timestamp del primer análisis (default: 2026-01-13_09-28-08)'
    )
    parser.add_argument(
        '--analysis2',
        default='2026-01-12_10-13-37',
        help='Timestamp del segundo análisis (default: 2026-01-12_10-13-37)'
    )
    parser.add_argument(
        '--name1',
        default='Sin interpolación (2026-01-13)',
        help='Nombre descriptivo del análisis 1'
    )
    parser.add_argument(
        '--name2',
        default='Con interpolación (2026-01-12)',
        help='Nombre descriptivo del análisis 2'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generar visualizaciones comprehensivas'
    )

    args = parser.parse_args()

    analysis1_path = Path(DATA_DIR) / "hand_analysis" / args.analysis1
    analysis2_path = Path(DATA_DIR) / "hand_analysis" / args.analysis2

    name1 = args.name1
    name2 = args.name2

    print("="*80)
    print("COMPARACIÓN DE ANÁLISIS: INTERPOLACIÓN")
    print("="*80)
    print(f"Análisis 1: {name1}")
    print(f"Análisis 2: {name2}")
    
    # Cargar datos
    print("\nCargando datos...")
    df1 = load_analysis_data(analysis1_path)
    df2 = load_analysis_data(analysis2_path)
    print(f"  {name1}: {len(df1):,} trials cargados")
    print(f"  {name2}: {len(df2):,} trials cargados")
    
    # 1. Comparación estructural
    common_cols, only_in_1, only_in_2 = compare_structure(df1, df2, name1, name2)
    
    # 2. Comparación de métricas numéricas
    differences_df = compare_numeric_metrics(df1, df2, common_cols, name1, name2)
    
    # 3. Comparación por sujeto
    common_subjects, only_in_1_subj, only_in_2_subj = compare_by_subject(df1, df2, name1, name2)
    
    # 4. Análisis profundo (solo si hay diferencias)
    if len(differences_df) > 0:
        deep_analysis_differences(df1, df2, differences_df, common_cols, name1, name2)
    
    # 5. Reporte final
    output_path = Path(DATA_DIR) / "comparison_interpolation_report.txt"
    generate_report(df1, df2, differences_df, common_subjects, name1, name2, output_path)

    # 6. Visualizaciones (si se solicitan)
    if args.visualize:
        print("\n" + "="*80)
        print("GENERANDO VISUALIZACIONES")
        print("="*80)
        output_dir = Path(DATA_DIR) / "comparison_figures"
        figure_paths = create_visualizations(
            df1, df2, differences_df, common_cols, name1, name2, output_dir
        )
        print(f"\n📊 {len(figure_paths)} figuras guardadas en: {output_dir}")
        for fig_path in figure_paths:
            print(f"   - {fig_path.name}")

    print("\n" + "="*80)
    print("COMPARACIÓN COMPLETADA")
    print("="*80)


if __name__ == "__main__":
    main()
