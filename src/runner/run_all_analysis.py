#!/usr/bin/env python3
"""
Ejecutar todos los análisis de experimentos neuropsicológicos.

Tareas:
- TMT (Trail Making Test)
- SST (Stop Signal Task)
- CDT (Change Detection Task)
- Go/No-Go

Uso:
    python -m src.runner.run_all_analysis
"""
import logging
from datetime import datetime

from src.config import RANDOM_STATE
from src.loader import load_sst_analysis, load_cdt_analysis, load_gonogo_analysis
from src.loader.analysis_loader import load_analysis as load_tmt_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    """Ejecutar todos los análisis y mostrar resumen."""
    print("=" * 60)
    print(f"ANÁLISIS DE EXPERIMENTOS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # TMT (Trail Making Test)
    print("\n🧠 Analizando TMT...")
    tmt_df, tmt_path = load_tmt_analysis(RANDOM_STATE, 1, False, None)
    results["TMT"] = {"n_subjects": len(tmt_df), "path": tmt_path}
    print(f"   → {len(tmt_df)} sujetos analizados")

    # SST (Stop Signal Task)
    print("\n🛑 Analizando SST...")
    sst_df, sst_path = load_sst_analysis(save_results=True)
    results["SST"] = {"n_subjects": len(sst_df), "path": sst_path}
    print(f"   → {len(sst_df)} sujetos analizados")

    # CDT (Change Detection Task)
    print("\n🔲 Analizando CDT...")
    cdt_df, cdt_path = load_cdt_analysis(save_results=True)
    results["CDT"] = {"n_subjects": len(cdt_df), "path": cdt_path}
    print(f"   → {len(cdt_df)} sujetos analizados")

    # Go/No-Go
    print("\n🚦 Analizando Go/No-Go...")
    gonogo_df, gonogo_path = load_gonogo_analysis(save_results=True)
    results["Go/No-Go"] = {"n_subjects": len(gonogo_df), "path": gonogo_path}
    print(f"   → {len(gonogo_df)} sujetos analizados")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for task, info in results.items():
        print(f"  {task:10} | {info['n_subjects']:4} sujetos | {info['path']}")
    print("=" * 60)
    print("\n✅ Todos los análisis completados!")


if __name__ == "__main__":
    main()

