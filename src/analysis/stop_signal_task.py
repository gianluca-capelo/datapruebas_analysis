import logging
from decimal import Decimal, ROUND_HALF_EVEN
import numpy as np
import pandas as pd

# --- CONSTANTES DE CALIDAD ---
MIN_RT_THRESHOLD = 150      # ms
MIN_GO_ACCURACY = 0.60      # 60%
MIN_PRESP = 0.10            # 10%
MAX_PRESP = 0.90            # 90%
# -----------------------------

def round_half_even(value):
    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_EVEN))


class StopSignalTask:

    def run(self, subjects_data):
        all_metrics = pd.DataFrame()
        invalid_subjects = []
        
        for subject, subject_df in subjects_data.items():
            print(f"Processing subject: {subject}")

            if all(subject_df['response'] == 'undefined'):
                logging.warning(f'All responses are undefined, skipping subject {subject}')
                invalid_subjects.append(subject)
                continue

            datos = subject_df

            # Filter out training data
            try:
                datos_custom_plugin = datos[datos['block_i'] > 0].copy()
            except KeyError:
                logging.warning(f'No block_i column found, skipping subject {subject}')
                invalid_subjects.append(subject)
                continue

            # --- CHECK 1: LIMPIEZA DE ANTICIPACIONES (< 150ms) ---
            # 1. Convertimos 'rt' a numérico. "undefined" se convierte en NaN.
            datos_custom_plugin['rt'] = pd.to_numeric(datos_custom_plugin['rt'], errors='coerce')
            
            # 2. Filtramos. Mantenemos si rt >= umbral O si rt es NaN (omisión/stop exitoso)
            mask_valid_rt = (datos_custom_plugin['rt'] >= MIN_RT_THRESHOLD) | (datos_custom_plugin['rt'].isna())
            datos_clean = datos_custom_plugin[mask_valid_rt]
            # -----------------------------------------------------

            # Stop-signal trials
            stopsignal = datos_clean[datos_clean['signal'] == 'yes']
            Nstop = len(stopsignal)

            # Cálculo de presp (CORREGIDO)
            # Usamos 'rt' para chequear respuesta porque 'response' tiene strings "undefined"
            tmp = np.where(stopsignal['rt'].isna(), 0, 1)
            presp = np.mean(tmp)
            print(f'presp: {presp}')

            # --- CHECK 2: VALIDACIÓN DE PROBABILIDAD DE RESPUESTA ---
            if presp < MIN_PRESP or presp > MAX_PRESP:
                logging.warning(f'Subject {subject} excluded: Invalid presp ({presp:.2f}). Staircase failed.')
                invalid_subjects.append(subject)
                continue
            # --------------------------------------------------------

            ssd = round_half_even(stopsignal['SSD'].mean())
            print(f'ssd: {ssd}')

            # Filtrar trials con respuesta para calcular usRT (CORREGIDO)
            stopsignal_resp_trials = stopsignal[stopsignal['rt'].notna()]
            
            usRTtmp = stopsignal_resp_trials['rt']
            usRTtmp_mean = np.nanmean(usRTtmp)
            usRT = np.nan if np.isnan(usRTtmp_mean) else round_half_even(usRTtmp_mean)
            print(f'usRT: {usRT}')

            # Go trials
            go = datos_clean[datos_clean['signal'] == 'no']
            Ngo = len(go)

            # Filtrar trials con respuesta para Go (CORREGIDO)
            go_resp_trials = go[go['rt'].notna()]
            
            # --- CHECK 3: PRECISIÓN EN GO (ACCURACY) ---
            go_correct_trials = go[go['correct'] == True]
            if len(go_resp_trials) > 0:
                acc = len(go_correct_trials) / len(go_resp_trials)
            else:
                acc = 0
            
            if acc < MIN_GO_ACCURACY:
                logging.warning(f'Subject {subject} excluded: Low Go accuracy ({acc:.2f})')
                invalid_subjects.append(subject)
                continue
            # -------------------------------------------

            goRTtmp = go_resp_trials['rt']
            goRTtmp_mean = np.nanmean(goRTtmp)
            goRT_all = np.nan if np.isnan(goRTtmp_mean) else round_half_even(goRTtmp_mean)
            goRTtmp_sd = np.nanstd(goRTtmp, ddof=1)
            goRT_sd = np.nan if np.isnan(goRTtmp_sd) else round_half_even(goRTtmp_sd)

            # Lógica para Nth (CORREGIDO)
            goRT_max = go_resp_trials['rt'].max()
            
            # Si rt es NaN (omisión), usamos goRT_max
            goRT_adj = np.where(go['rt'].isna(), goRT_max, go['rt'])
            
            quantile = np.quantile(goRT_adj, presp, method='weibull')
            nth = np.nan if np.isnan(quantile) else round_half_even(quantile)
            print(f'nth: {nth}')

            go_correct_trials_mean = np.nanmean(go_correct_trials['rt'])
            goRT_correct = np.nan if np.isnan(go_correct_trials_mean) else round_half_even(go_correct_trials_mean)

            go_omission = 1 - (len(go_resp_trials) / Ngo)
            
            if len(go_resp_trials) > 0:
                go_error = 1 - (len(go_correct_trials) / len(go_resp_trials))
            else:
                go_error = 1.0

            go_premature = 0 

            # Calculate SSRT
            ssrt = nth - ssd
            print(f'ssrt: {ssrt}')
            print('-------------------------------------')

            # --- CHECK 4: SSRT NEGATIVO ---
            if ssrt < 0:
                logging.warning(f'Subject {subject} excluded: Negative SSRT ({ssrt}).')
                invalid_subjects.append(subject)
                continue
            # ------------------------------

            metrics = pd.DataFrame({
                'Subject': [subject],
                'Nstop': [Nstop],
                'presp': [presp],
                'ssd': [ssd],
                'usRT': [usRT],
                'goRT_all': [goRT_all],
                'goRT_sd': [goRT_sd],
                'nth': [nth],
                'goRT_correct': [goRT_correct],
                'go_omission': [go_omission],
                'go_error': [go_error],
                'go_premature': [go_premature],
                'ssrt': [ssrt]
            })

            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)

        print(f"Skipped {len(invalid_subjects)} subjects: {invalid_subjects}")

        return all_metrics