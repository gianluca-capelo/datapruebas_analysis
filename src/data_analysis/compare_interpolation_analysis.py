#!/usr/bin/env python3
"""
Comparar análisis sin interpolación vs con interpolación.

Compara:
- Análisis 1: Sin interpolación (INTERPOLATE_TRAJECTORY=False)
- Análisis 2: Con interpolación (INTERPOLATE_TRAJECTORY=True)

Uso:
    python -m src.data_analysis.compare_interpolation_analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path


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


def main():
    """Función principal."""
    from src.config import DATA_DIR
    
    analysis1_path = Path(DATA_DIR) / "hand_analysis" / "2026-01-13_09-28-08"
    analysis2_path = Path(DATA_DIR) / "hand_analysis" / "2026-01-12_10-13-37"
    
    name1 = "Sin interpolación (2026-01-13)"
    name2 = "Con interpolación (2026-01-12)"
    
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
    
    print("\n" + "="*80)
    print("COMPARACIÓN COMPLETADA")
    print("="*80)


if __name__ == "__main__":
    main()
