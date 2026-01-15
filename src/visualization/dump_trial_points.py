#!/usr/bin/env python
"""
Script para extraer y guardar los puntos de un trial.
Ejecutar dos veces con diferentes valores de INTERPOLATE_TRAJECTORY.

Uso:
    # Con INTERPOLATE_TRAJECTORY = False en config.py
    python -m src.visualization.dump_trial_points --subject "UUID" --trial DATAPRUEBAS_0 --output points_no_interp.txt

    # Cambiar config.py a INTERPOLATE_TRAJECTORY = True
    python -m src.visualization.dump_trial_points --subject "UUID" --trial DATAPRUEBAS_0 --output points_interp.txt

    # Comparar
    diff points_no_interp.txt points_interp.txt
"""

import argparse
import os
from src import config
from src.visualization.plot_trials_cli import load_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--origin", default="datapruebas")
    args = parser.parse_args()

    print(f"Config INTERPOLATE_TRAJECTORY = {config.INTERPOLATE_TRAJECTORY}")

    experiment = load_experiment(origin=args.origin)
    subject = experiment.subjects[args.subject]

    for trial in subject.testing_trials:
        if trial.id == args.trial:
            cursor_trail = trial.get_cursor_trail_from_start()

            with open(args.output, 'w') as f:
                f.write(f"# INTERPOLATE_TRAJECTORY = {config.INTERPOLATE_TRAJECTORY}\n")
                f.write(f"# Total points: {len(cursor_trail)}\n")
                f.write("# x, y, t\n")
                for p in cursor_trail:
                    f.write(f"{p.position.x}, {p.position.y}, {p.time}\n")

            print(f"Guardado {len(cursor_trail)} puntos en {args.output}")
            return

    print(f"Trial {args.trial} no encontrado")


if __name__ == "__main__":
    main()
