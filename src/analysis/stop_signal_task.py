import logging
from decimal import Decimal, ROUND_HALF_EVEN

import numpy as np
import pandas as pd


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
                datos_custom_plugin = datos[datos['block_i'] > 0]
            except KeyError:
                logging.warning(f'No block_i column found, skipping subject {subject}')
                invalid_subjects.append(subject)
                continue

            # Stop-signal trials
            stopsignal = datos_custom_plugin[datos_custom_plugin['signal'] == 'yes']
            Nstop = len(stopsignal)

            tmp = np.where(stopsignal['response'] == 'undefined', 0, 1)
            presp = np.mean(tmp)
            print('presp:')
            print(presp)

            ssd = round_half_even(stopsignal['SSD'].mean())
            print('ssd:')
            print(ssd)

            stopsignal_resp_trials = stopsignal[stopsignal['response'] != 'undefined']
            usRTtmp = stopsignal_resp_trials['rt'].fillna(-250)
            usRTtmp_mean = np.nanmean(usRTtmp)
            usRT = np.nan if np.isnan(usRTtmp_mean) else round_half_even(usRTtmp_mean)
            print('usRT:')
            print(usRT)

            # Go trials
            go = datos_custom_plugin[datos_custom_plugin['signal'] == 'no']
            Ngo = len(go)

            go_resp_trials = go[go['response'] != 'undefined']
            goRTtmp = go_resp_trials['rt'].fillna(-250)
            goRTtmp_mean = np.nanmean(goRTtmp)
            goRT_all = np.nan if np.isnan(goRTtmp_mean) else round_half_even(goRTtmp_mean)
            goRTtmp_sd = np.nanstd(goRTtmp, ddof=1)
            goRT_sd = np.nan if np.isnan(goRTtmp_sd) else round_half_even(goRTtmp_sd)

            goRT_max = go_resp_trials['rt'].max()
            goRT_adj = np.where(go['response'] == 'undefined', goRT_max, go['rt'])
            quantile = np.quantile(goRT_adj, presp, method='weibull')
            nth = np.nan if np.isnan(quantile) else round_half_even(quantile)
            print('nth:')
            print(nth)

            go_correct_trials = go[go['correct'] == True]
            go_correct_trials_mean = np.nanmean(go_correct_trials['rt'])
            goRT_correct = np.nan if np.isnan(go_correct_trials_mean) else round_half_even(go_correct_trials_mean)

            go_omission = 1 - (len(go_resp_trials) / Ngo)
            try:
                go_error = 1 - (len(go_correct_trials) / len(go_resp_trials))
            except ZeroDivisionError:
                logging.warning(f'No go trials found, skipping subject {subject}')
                invalid_subjects.append(subject)
                continue

            premature_trials = go[go['rt'] < 0]
            go_premature = len(premature_trials) / Ngo

            # Calculate SSRT
            ssrt = nth - ssd
            print('ssrt:')
            print(ssrt)
            print('-------------------------------------')

            # Collect all metrics in a DataFrame
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

            # Combine all metrics for all subjects
            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)

        print(f"Skipped {len(invalid_subjects)} subjects")

        return all_metrics

