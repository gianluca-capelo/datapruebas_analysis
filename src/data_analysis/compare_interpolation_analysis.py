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
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional


def load_config(analysis_path: Path) -> Dict[str, Any]:
    """Cargar configuration.json de un análisis."""
    config_path = Path(analysis_path) / "configuration.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def validate_configs(path1: Path, path2: Path, name1: str, name2: str) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """
    Validar que ambos análisis tienen configuraciones compatibles.

    Returns:
        Tuple de (is_valid, config1, config2)
    """
    print("\n" + "="*80)
    print("0. VALIDACIÓN DE CONFIGURACIONES")
    print("="*80)

    config1 = load_config(path1)
    config2 = load_config(path2)

    if not config1 or not config2:
        print("WARNING: No se pudo cargar configuration.json de uno o ambos análisis")
        if not config1:
            print(f"  - Falta configuration.json en {path1}")
        if not config2:
            print(f"  - Falta configuration.json en {path2}")
        return False, config1, config2

    # Campos que DEBEN ser iguales para una comparación válida
    must_match = [
        'correct_targets_minimum',
        'cut_criteria',
        'consecutive_points',
        'calculate_crosses'
    ]

    # Campos que esperamos que difieran
    expected_different = ['interpolate_trajectory', 'timestamp', 'git_commit', 'git_dirty']

    all_match = True
    print(f"\nVerificando configuraciones:")
    print(f"{'Campo':<30} {name1:<20} {name2:<20} {'Status':<15}")
    print("-"*85)

    for field in must_match:
        val1 = config1.get(field, 'N/A')
        val2 = config2.get(field, 'N/A')
        match = val1 == val2
        status = "OK" if match else "DIFERENTE"
        if not match:
            all_match = False
        print(f"{field:<30} {str(val1):<20} {str(val2):<20} {status:<15}")

    # Verificar interpolate_trajectory
    interp1 = config1.get('interpolate_trajectory', 'NO GUARDADO')
    interp2 = config2.get('interpolate_trajectory', 'NO GUARDADO')

    print(f"\n{'interpolate_trajectory':<30} {str(interp1):<20} {str(interp2):<20}")

    if interp1 == 'NO GUARDADO' or interp2 == 'NO GUARDADO':
        print("\nWARNING: 'interpolate_trajectory' no está guardado en configuration.json")
        print("No se puede verificar que solo difiere la interpolación!")
        print("Considera regenerar los análisis para tener certeza.")
    elif interp1 == interp2:
        print("\nWARNING: Ambos análisis tienen el mismo valor de interpolate_trajectory!")
        print("La comparación puede no ser válida para evaluar el efecto de interpolación.")
    else:
        print("\nOK: Los análisis difieren en interpolate_trajectory (esperado)")

    if not all_match:
        print("\nERROR: Los análisis tienen configuraciones diferentes en campos críticos.")
        print("La comparación puede no ser válida.")

    return all_match, config1, config2


def compare_by_trial_id(df1: pd.DataFrame, df2: pd.DataFrame,
                         name1: str, name2: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Comparar análisis trial por trial usando merge por (subject_id, trial_id).

    Returns:
        Tuple de (merged_df, trial_differences_df)
    """
    print("\n" + "="*80)
    print("COMPARACIÓN TRIAL POR TRIAL")
    print("="*80)

    # Verificar que existen las columnas necesarias
    if 'subject_id' not in df1.columns or 'trial_id' not in df1.columns:
        print("ERROR: df1 no tiene columnas subject_id o trial_id")
        return pd.DataFrame(), pd.DataFrame()

    if 'subject_id' not in df2.columns or 'trial_id' not in df2.columns:
        print("ERROR: df2 no tiene columnas subject_id o trial_id")
        return pd.DataFrame(), pd.DataFrame()

    # Crear keys únicas
    trials1 = set(zip(df1['subject_id'], df1['trial_id']))
    trials2 = set(zip(df2['subject_id'], df2['trial_id']))

    common_trials = trials1 & trials2
    only_in_1 = trials1 - trials2
    only_in_2 = trials2 - trials1

    print(f"\nTrials en {name1}: {len(trials1):,}")
    print(f"Trials en {name2}: {len(trials2):,}")
    print(f"Trials comunes (matched): {len(common_trials):,}")
    print(f"Solo en {name1}: {len(only_in_1):,}")
    print(f"Solo en {name2}: {len(only_in_2):,}")

    coverage = len(common_trials) / min(len(trials1), len(trials2)) * 100
    print(f"\nCobertura del match: {coverage:.1f}%")

    if coverage < 95:
        print("WARNING: Menos del 95% de trials hacen match!")
        print("Esto puede indicar filtrado diferente entre análisis.")

    if only_in_1:
        print(f"\nTrials solo en {name1} (primeros 5):")
        for subj, trial in list(only_in_1)[:5]:
            print(f"  - {subj}, trial {trial}")

    if only_in_2:
        print(f"\nTrials solo en {name2} (primeros 5):")
        for subj, trial in list(only_in_2)[:5]:
            print(f"  - {subj}, trial {trial}")

    # Hacer merge
    merged = df1.merge(
        df2,
        on=['subject_id', 'trial_id'],
        suffixes=('_base', '_interp'),
        how='inner'
    )

    print(f"\nTrials después del merge: {len(merged):,}")

    return merged, pd.DataFrame()  # trial_differences se calcula en otra función


def calculate_trial_differences(merged_df: pd.DataFrame,
                                 metric_cols: list) -> pd.DataFrame:
    """
    Calcular diferencias trial por trial para métricas específicas.

    Args:
        merged_df: DataFrame con columnas _base y _interp
        metric_cols: Lista de nombres de métricas (sin sufijo)

    Returns:
        DataFrame con estadísticas de diferencias por métrica
    """
    results = []

    for col in metric_cols:
        col_base = f"{col}_base"
        col_interp = f"{col}_interp"

        if col_base not in merged_df.columns or col_interp not in merged_df.columns:
            continue

        # Saltar columnas booleanas
        if merged_df[col_base].dtype == bool or merged_df[col_interp].dtype == bool:
            continue

        # Convertir a numérico
        base = pd.to_numeric(merged_df[col_base], errors='coerce')
        interp = pd.to_numeric(merged_df[col_interp], errors='coerce')

        # Calcular diferencias solo donde ambos tienen valores
        mask = base.notna() & interp.notna()
        if mask.sum() == 0:
            continue

        diff = interp[mask] - base[mask]
        diff_pct = (diff / base[mask].replace(0, np.nan)) * 100

        results.append({
            'metric': col,
            'n_trials': mask.sum(),
            'mean_diff': diff.mean(),
            'median_diff': diff.median(),
            'std_diff': diff.std(),
            'mean_diff_pct': diff_pct.mean(),
            'median_diff_pct': diff_pct.median(),
            'p5_diff': diff.quantile(0.05),
            'p95_diff': diff.quantile(0.95),
            'p5_diff_pct': diff_pct.quantile(0.05),
            'p95_diff_pct': diff_pct.quantile(0.95),
            'max_abs_diff': diff.abs().max(),
            'max_abs_diff_pct': diff_pct.abs().max()
        })

    return pd.DataFrame(results)


def find_problematic_trials(merged_df: pd.DataFrame, metric: str,
                            top_n: int = 10) -> pd.DataFrame:
    """
    Encontrar los trials con mayor diferencia para una métrica específica.
    """
    col_base = f"{metric}_base"
    col_interp = f"{metric}_interp"

    if col_base not in merged_df.columns or col_interp not in merged_df.columns:
        return pd.DataFrame()

    df = merged_df[['subject_id', 'trial_id', col_base, col_interp]].copy()
    df['diff'] = pd.to_numeric(df[col_interp], errors='coerce') - pd.to_numeric(df[col_base], errors='coerce')
    df['diff_pct'] = (df['diff'] / pd.to_numeric(df[col_base], errors='coerce').replace(0, np.nan)) * 100
    df['abs_diff_pct'] = df['diff_pct'].abs()

    df = df.dropna(subset=['diff'])

    return df.nlargest(top_n, 'abs_diff_pct')[['subject_id', 'trial_id', col_base, col_interp, 'diff', 'diff_pct']]


def create_bland_altman_plot(merged_df: pd.DataFrame, metric: str,
                              name1: str, name2: str, output_dir: Path) -> Optional[Path]:
    """
    Crear Bland-Altman plot para una métrica.

    Bland-Altman plot muestra:
    - X: promedio de ambas mediciones
    - Y: diferencia entre mediciones
    - Líneas horizontales en mean ± 1.96*std (límites de acuerdo del 95%)
    """
    col_base = f"{metric}_base"
    col_interp = f"{metric}_interp"

    if col_base not in merged_df.columns or col_interp not in merged_df.columns:
        return None

    base = pd.to_numeric(merged_df[col_base], errors='coerce')
    interp = pd.to_numeric(merged_df[col_interp], errors='coerce')

    mask = base.notna() & interp.notna()
    if mask.sum() < 10:
        return None

    base = base[mask]
    interp = interp[mask]

    mean_vals = (base + interp) / 2
    diff_vals = interp - base

    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=20)

    # Línea de media
    ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')

    # Límites de acuerdo (±1.96 SD)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1, label=f'+1.96 SD: {upper_limit:.3f}')
    ax.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1, label=f'-1.96 SD: {lower_limit:.3f}')

    # Línea en y=0
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel(f'Promedio ({name1} + {name2}) / 2', fontsize=11)
    ax.set_ylabel(f'Diferencia ({name2} - {name1})', fontsize=11)
    ax.set_title(f'Bland-Altman Plot: {metric}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / f"bland_altman_{metric.replace('/', '_')[:40]}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig_path


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
        default='Sin interpolación',
        help='Nombre descriptivo del análisis 1'
    )
    parser.add_argument(
        '--name2',
        default='Con interpolación',
        help='Nombre descriptivo del análisis 2'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generar visualizaciones comprehensivas'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Saltar validación de configuraciones'
    )

    args = parser.parse_args()

    analysis1_path = Path(DATA_DIR) / "hand_analysis" / args.analysis1
    analysis2_path = Path(DATA_DIR) / "hand_analysis" / args.analysis2

    name1 = args.name1
    name2 = args.name2

    print("="*80)
    print("COMPARACIÓN DE ANÁLISIS: INTERPOLACIÓN")
    print("="*80)
    print(f"Análisis 1: {name1} ({args.analysis1})")
    print(f"Análisis 2: {name2} ({args.analysis2})")

    # 0. Validación de configuraciones
    if not args.skip_validation:
        configs_valid, config1, config2 = validate_configs(analysis1_path, analysis2_path, name1, name2)
    else:
        print("\nSaltando validación de configuraciones...")
        configs_valid = True

    # Cargar datos
    print("\nCargando datos...")
    df1 = load_analysis_data(analysis1_path)
    df2 = load_analysis_data(analysis2_path)
    print(f"  {name1}: {len(df1):,} trials cargados")
    print(f"  {name2}: {len(df2):,} trials cargados")

    # 1. Comparación estructural
    common_cols, only_in_1, only_in_2 = compare_structure(df1, df2, name1, name2)

    # 2. Comparación de métricas numéricas (agregada)
    differences_df = compare_numeric_metrics(df1, df2, common_cols, name1, name2)

    # 3. Comparación por sujeto
    common_subjects, only_in_1_subj, only_in_2_subj = compare_by_subject(df1, df2, name1, name2)

    # 4. NUEVO: Comparación trial por trial
    merged_df, _ = compare_by_trial_id(df1, df2, name1, name2)

    # 5. NUEVO: Calcular diferencias trial por trial para métricas clave
    if len(merged_df) > 0:
        # Identificar métricas numéricas
        numeric_cols = []
        for col in common_cols:
            if col in ['subject_id', 'trial_id', 'experiment_origin', 'invalid_cause']:
                continue
            try:
                pd.to_numeric(df1[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue

        trial_diff_df = calculate_trial_differences(merged_df, numeric_cols)

        if len(trial_diff_df) > 0:
            print("\n" + "="*80)
            print("DIFERENCIAS TRIAL POR TRIAL (Top 20 métricas)")
            print("="*80)
            trial_diff_df['abs_mean_diff_pct'] = trial_diff_df['mean_diff_pct'].abs()
            trial_diff_df = trial_diff_df.sort_values('abs_mean_diff_pct', ascending=False)

            print(f"\n{'Métrica':<35} {'N':<8} {'Mean Δ%':<12} {'Median Δ%':<12} {'p5-p95 Δ%':<20}")
            print("-"*90)
            for _, row in trial_diff_df.head(20).iterrows():
                p5_p95 = f"[{row['p5_diff_pct']:.1f}, {row['p95_diff_pct']:.1f}]"
                print(f"{row['metric']:<35} {row['n_trials']:<8} {row['mean_diff_pct']:<12.2f} {row['median_diff_pct']:<12.2f} {p5_p95:<20}")

            # Mostrar trials problemáticos para las top 3 métricas con mayor diferencia
            print("\n" + "="*80)
            print("TRIALS CON MAYOR DIFERENCIA (Top 3 métricas)")
            print("="*80)

            for _, row in trial_diff_df.head(3).iterrows():
                metric = row['metric']
                print(f"\n--- {metric} (Mean Δ%: {row['mean_diff_pct']:.2f}%) ---")
                problematic = find_problematic_trials(merged_df, metric, top_n=5)
                if len(problematic) > 0:
                    print(f"{'Subject ID':<45} {'Trial':<8} {'Base':<12} {'Interp':<12} {'Δ%':<10}")
                    print("-"*90)
                    for _, trial_row in problematic.iterrows():
                        col_base = f"{metric}_base"
                        col_interp = f"{metric}_interp"
                        print(f"{trial_row['subject_id'][:44]:<45} {trial_row['trial_id']:<8} "
                              f"{trial_row[col_base]:<12.4f} {trial_row[col_interp]:<12.4f} "
                              f"{trial_row['diff_pct']:<10.2f}")

            # Guardar CSV con diferencias por métrica
            csv_output_path = Path(DATA_DIR) / "comparison_metrics_diff.csv"
            trial_diff_df.to_csv(csv_output_path, index=False)
            print(f"\nCSV de diferencias por métrica guardado en: {csv_output_path}")

    # 6. Análisis profundo (solo si hay diferencias)
    if len(differences_df) > 0:
        deep_analysis_differences(df1, df2, differences_df, common_cols, name1, name2)

    # 7. Reporte final
    output_path = Path(DATA_DIR) / "comparison_interpolation_report.txt"
    generate_report(df1, df2, differences_df, common_subjects, name1, name2, output_path)

    # 8. Visualizaciones (si se solicitan)
    if args.visualize:
        print("\n" + "="*80)
        print("GENERANDO VISUALIZACIONES")
        print("="*80)
        output_dir = Path(DATA_DIR) / "comparison_figures"
        figure_paths = create_visualizations(
            df1, df2, differences_df, common_cols, name1, name2, output_dir
        )

        # Generar Bland-Altman plots para las top métricas
        if len(merged_df) > 0 and len(trial_diff_df) > 0:
            print("\nGenerando Bland-Altman plots...")
            for metric in trial_diff_df.head(6)['metric'].tolist():
                ba_path = create_bland_altman_plot(merged_df, metric, name1, name2, output_dir)
                if ba_path:
                    figure_paths.append(ba_path)

        print(f"\n{len(figure_paths)} figuras guardadas en: {output_dir}")
        for fig_path in figure_paths:
            print(f"   - {fig_path.name}")

    print("\n" + "="*80)
    print("COMPARACIÓN COMPLETADA")
    print("="*80)


if __name__ == "__main__":
    main()
