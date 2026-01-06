import logging
from src.model.datasetbuilder.dataset_builder import DatasetBuilder

# Configurar log básico para ver mensajes claros
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("\n🚀 INICIANDO TEST DEL DATASET BUILDER")
    print("=" * 60)

    builder = DatasetBuilder()
    
    # Lista de datasets a probar según lo que definiste en tu clase
    datasets_to_test = ['tmt_ssrt', 'tmt_k', 'tmt_dprime']

    for ds_name in datasets_to_test:
        print(f"\n🔹 Intentando construir: '{ds_name}'")
        try:
            # Llamada al método que acabamos de crear
            X, y, feature_names, target_name = builder.get_dataset(ds_name)
            
            # Si no explota, mostramos el éxito y estadísticas básicas
            print("  ✅ ¡ÉXITO!")
            print(f"  📊 Dimensiones X: {X.shape} (Sujetos x Features)")
            print(f"  🎯 Dimensiones y: {y.shape} (Targets)")
            print(f"  🏷️  Nombre Target: '{target_name}'")
            print(f"  👀 Primeras 3 features: {feature_names[:3]}")
            
            # Verificación de sanidad: X e y deben tener mismo número de filas
            if X.shape[0] != y.shape[0]:
                print("  ⚠️  ALERTA: Mismatch en número de filas entre X e y")
            
        except RuntimeError as e:
            # Errores esperados si faltan datos (ej. "No SST analysis found")
            print(f"  🔸 NO SE PUDO CONSTRUIR (Causa esperada): {e}")
        except ValueError as e:
            # Errores de configuración (ej. columna no encontrada)
            print(f"  ❌ ERROR DE CONFIGURACIÓN: {e}")
        except Exception as e:
            # Otros errores inesperados
            print(f"  💥 ERROR INESPERADO: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()