import logging
import pytest
import pandas as pd
from src.model.datasetbuilder.dataset_builder import DatasetBuilder

# Configurar log basico para ver mensajes claros
logging.basicConfig(level=logging.INFO, format='%(message)s')


# =============================================================================
# Unit Tests (pytest)
# =============================================================================

class TestTrialTypeCoverageFilter:
    """Tests for the trial type coverage filter (PART_A and PART_B requirement)."""

    def test_excludes_subjects_without_both_trial_types(self):
        """Verify that subjects missing PART_A or PART_B are excluded."""
        builder = DatasetBuilder()

        # Create test DataFrame with incomplete subjects
        test_df = pd.DataFrame({
            'subject_id': ['S1', 'S1', 'S2', 'S2', 'S3'],
            'trial_type': ['PART_A', 'PART_B', 'PART_A', 'PART_A', 'PART_B'],
            'is_valid': ['True', 'True', 'True', 'True', 'True'],
            'rt': [100, 200, 150, 160, 250]
        })

        valid_df = builder._get_valid_tmt_trials(test_df)
        result = builder._aggregate_tmt(valid_df)

        # S1 has both types -> included
        # S2 only has PART_A -> excluded
        # S3 only has PART_B -> excluded
        assert 'S1' in result['subject_id'].values
        assert 'S2' not in result['subject_id'].values
        assert 'S3' not in result['subject_id'].values
        assert len(result) == 1

    def test_includes_subjects_with_multiple_trials_of_each_type(self):
        """Verify subjects with multiple trials of each type are included."""
        builder = DatasetBuilder()

        test_df = pd.DataFrame({
            'subject_id': ['S1', 'S1', 'S1', 'S1'],
            'trial_type': ['PART_A', 'PART_A', 'PART_B', 'PART_B'],
            'is_valid': ['True', 'True', 'True', 'True'],
            'rt': [100, 110, 200, 210]
        })

        valid_df = builder._get_valid_tmt_trials(test_df)
        result = builder._aggregate_tmt(valid_df)

        assert 'S1' in result['subject_id'].values
        assert len(result) == 1
        # Check that mean was computed correctly
        assert 'rt_PART_A' in result.columns
        assert 'rt_PART_B' in result.columns

    def test_handles_invalid_trials_correctly(self):
        """Verify that invalid trials don't count toward coverage."""
        builder = DatasetBuilder()

        test_df = pd.DataFrame({
            'subject_id': ['S1', 'S1', 'S1', 'S2', 'S2'],
            'trial_type': ['PART_A', 'PART_B', 'PART_B', 'PART_A', 'PART_B'],
            'is_valid': ['True', 'True', 'False', 'True', 'False'],  # S2's PART_B is invalid
            'rt': [100, 200, 300, 150, 250]
        })

        valid_df = builder._get_valid_tmt_trials(test_df)
        result = builder._aggregate_tmt(valid_df)

        # S1 has valid PART_A and PART_B -> included
        # S2 has valid PART_A but no valid PART_B -> excluded
        assert 'S1' in result['subject_id'].values
        assert 'S2' not in result['subject_id'].values

    def test_raises_assertion_when_no_subjects_remain(self):
        """Verify assertion error when all subjects are excluded."""
        builder = DatasetBuilder()

        test_df = pd.DataFrame({
            'subject_id': ['S1', 'S2'],
            'trial_type': ['PART_A', 'PART_B'],
            'is_valid': ['True', 'True'],
            'rt': [100, 200]
        })

        with pytest.raises(AssertionError, match="No subjects remain"):
            builder._get_valid_tmt_trials(test_df)

    def test_exclusion_report_returns_correct_structure(self):
        """Verify get_exclusion_report returns expected dictionary structure."""
        builder = DatasetBuilder()

        test_df = pd.DataFrame({
            'subject_id': ['S1', 'S1', 'S2', 'S3'],
            'trial_type': ['PART_A', 'PART_B', 'PART_A', 'PART_B'],
            'is_valid': ['True', 'True', 'True', 'True'],
            'rt': [100, 200, 150, 250]
        })

        report = builder.get_exclusion_report(test_df)

        assert 'total_subjects' in report
        assert 'valid_subjects' in report
        assert 'excluded_subjects' in report
        assert 'exclusion_reasons' in report

        assert report['total_subjects'] == 3
        assert report['valid_subjects'] == 1
        assert 'S2' in report['excluded_subjects']
        assert 'S3' in report['excluded_subjects']
        assert 'S1' not in report['excluded_subjects']

    def test_exclusion_report_reasons_are_descriptive(self):
        """Verify exclusion reasons clearly indicate the missing type."""
        builder = DatasetBuilder()

        test_df = pd.DataFrame({
            'subject_id': ['S1', 'S2'],
            'trial_type': ['PART_A', 'PART_B'],
            'is_valid': ['True', 'True'],
            'rt': [100, 200]
        })

        report = builder.get_exclusion_report(test_df)

        assert 'Missing PART_B' in report['exclusion_reasons']['S1']
        assert 'Missing PART_A' in report['exclusion_reasons']['S2']


# =============================================================================
# Integration Test (manual execution)
# =============================================================================

def main():
    print("\nINICIANDO TEST DEL DATASET BUILDER")
    print("=" * 60)

    builder = DatasetBuilder()

    # Lista de datasets a probar segun lo que definiste en tu clase
    datasets_to_test = ['tmt_ssrt', 'tmt_k', 'tmt_dprime']

    for ds_name in datasets_to_test:
        print(f"\nIntentando construir: '{ds_name}'")
        try:
            # Llamada al metodo que acabamos de crear
            X, y, feature_names, target_name = builder.get_dataset(ds_name)

            # Si no explota, mostramos el exito y estadisticas basicas
            print("  EXITO!")
            print(f"  Dimensiones X: {X.shape} (Sujetos x Features)")
            print(f"  Dimensiones y: {y.shape} (Targets)")
            print(f"  Nombre Target: '{target_name}'")
            print(f"  Primeras 3 features: {feature_names[:3]}")

            # Verificacion de sanidad: X e y deben tener mismo numero de filas
            if X.shape[0] != y.shape[0]:
                print("  ALERTA: Mismatch en numero de filas entre X e y")

        except RuntimeError as e:
            # Errores esperados si faltan datos (ej. "No SST analysis found")
            print(f"  NO SE PUDO CONSTRUIR (Causa esperada): {e}")
        except ValueError as e:
            # Errores de configuracion (ej. columna no encontrada)
            print(f"  ERROR DE CONFIGURACION: {e}")
        except Exception as e:
            # Otros errores inesperados
            print(f"  ERROR INESPERADO: {e}")

        print("-" * 60)

if __name__ == "__main__":
    main()