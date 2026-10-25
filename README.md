# MMAI-Mortgage: code and results for the multimodal mortgage risk study

Supplementary material for the manuscript *"A Multimodal Artificial Intelligence Framework for Intelligent Mortgage Risk Assessment"* (COMPUTICS-2026-S53, revision). The manuscript title is being revised; the modality previously called "alternative" is the **behavioural / contextual (loan-performance) modality derived from the same Fannie Mae file**, not an independently acquired data source.

This package contains the complete pipeline, the exact outputs behind every table and figure of the paper, and a small script for the corrected statistics.

---

## 1. Contents

| File / folder | Purpose |
|---|---|
| `mmai_mortgage_pipeline.py` | Main pipeline: data loading, cohort construction, splits, preprocessing, all models, ablation, evaluation, SHAP, subgroup audit, figures and tables. |
| `corrected_statistics.py` | Post-hoc statistics: corrected resampled t-test (Nadeau and Bengio) with Holm adjustment and corrected confidence intervals. Reads only the pipeline outputs. |
| `results_v4/` | Frozen outputs of the final run reported in the paper (see Section 5). |
| `results_v4/stats_corrected/` | Output of `corrected_statistics.py`. |
| `requirements.txt` | Python dependencies. |

---

## 2. Data (not included)

The study uses the public **Fannie Mae Single-Family Loan Performance Data** (combined acquisition and performance files), available after free registration at <https://capitalmarkets.fanniemae.com>.

| Use | Acquisition quarter | File name expected by the script |
|---|---|---|
| Training, validation, internal test, later-age snapshot | 2007Q1 | `2007Q1.parquet` (110 columns; the pipe-separated `2007Q1.csv` also works, see below) |
| External test only (never used for training, tuning or thresholds) | 2018Q1 | `2018Q1.csv` (pipe-separated, no header) |

Place both files in the same folder as `mmai_mortgage_pipeline.py`. To read the original text file for 2007Q1, set `DATA_FILE = "2007Q1.csv"` in the settings block at the top of the script. The script reads the files in chunks and checks the column mapping at start-up (it stops with an error if the layout looks wrong).

The data are not redistributed here because of their size and the provider's terms.

---

## 3. Environment

The reported results were produced with:

| Component | Version |
|---|---|
| Python | 3.11.8 |
| PyTorch | 2.7.0+cu118 (CUDA GPU) |
| scikit-learn | 1.6.1 |
| XGBoost | 3.0.5 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| Platform | Windows 10, 16 CPU threads |

Also required: SciPy, Matplotlib, SHAP, PyArrow and tabulate (versions are not pinned; see `requirements.txt`).

```
pip install -r requirements.txt
```

---

## 4. How to run

### 4.1 Main pipeline

```
python mmai_mortgage_pipeline.py
```

No arguments. All settings are constants in the block at the top of the script. The values used for the paper are:

| Setting | Value | Meaning |
|---|---|---|
| `OBS_MONTHS` | 12 | Months of history; prediction date t0 = first reporting month + 12 |
| `HORIZON` | 12 | Months after t0 in which the outcome is observed |
| `DQ_THRESHOLD` | 3 | Delinquency status of 3 or more (90+ days) |
| `RUNS` | 10 | Repeated random splits, each with a new initialisation (seed = run index) |
| `N_TRIALS` | 6 | Hyper-parameter configurations per model (first = default) |
| `EPOCHS` | 60 | Maximum epochs for neural models (early stopping, patience 8, on validation PR-AUC) |
| `TRAIN_BALANCE` | `"undersample"` | Training set balanced 1:1 by random undersampling; validation and test sets keep the natural rate |
| `BOOT` | 500 | Bootstrap resamples (200 for the external bootstrap) |
| `SHAP_SAMPLE` | 1000 | Test loans used for SHAP |
| `EXTERNAL_TEST_FILE` | `"2018Q1.csv"` | External test file |

Everything is written to the folder `results/` (the folder was renamed `results_v4` for this package). The full run took about 34 minutes on the machine above (data loading about 6.5 minutes, tuning about 1.5 minutes, the three protocols about 12 minutes in total, SHAP and the imbalance study the rest).

### 4.2 Corrected statistics

```
python corrected_statistics.py
```

No arguments. If your results folder is not `D:\Amit-USA\Impaxon\results_v4`, edit `RESULTS_FOLDER` at the top of the script (it also falls back to a `results_v4` folder next to the script). The script uses only NumPy, pandas and SciPy, needs no data, uses no random numbers and gives identical output on every run.

---

## 5. Frozen results (`results_v4/`)

`results_v4/` holds the single complete execution reported in the paper. **Do not rerun the main pipeline if you want to compare against the paper**: neural-network training on a GPU is not bit-wise deterministic, so a fresh run can differ slightly (third decimal) from the reported values. The file `results_all_runs.csv` contains every per-run metric and is the source for all tables.

### 5.1 Paper tables and the files that produce them

| Paper item | File in `results_v4/` |
|---|---|
| Table 2 (features and modalities) | `feature_list.csv` |
| Table 3 (cohort and split sizes) | `run_config.json` (key `split_info`), `dataset_summary.csv` |
| Table 4 (baseline comparison, random split) | `table3_main_comparison_random_split.csv` (the file numbering of the pipeline differs from the paper) |
| Table 5 (paired comparisons) | `stats_corrected/table_stat_tests_random.csv` (corrected); `table_stat_tests_across_runs.csv` (original, uncorrected) |
| Table 6 (modality and fusion ablation) | `table4_ablation_random_split.csv` |
| Table 7 (later-age snapshot) | `table_out_of_time_validation.csv`; per-run top-5% recall and lift in `results_all_runs.csv` |
| Table 8 (external 2018Q1) | `table_external_validation_2018Q1.csv`, `table_external_top5pct.csv` |
| Table 9 (confusion matrix, run 0) | `table_confusion_and_classwise_run0.csv` |
| Table 10 (calibration and cost) | `table_cost_sensitive.csv`; Brier and ECE columns in `results_all_runs.csv` |
| Table 11 (class imbalance) | `table_class_imbalance_study.csv` |
| Table 13 (subgroup audit) | `subgroup_audit.csv` |
| Corrected confidence intervals | `stats_corrected/table_ci_random.csv`, `table_ci_oot.csv`, `table_ci_external.csv` |
| Corrected tests (later-age snapshot, external) | `stats_corrected/table_stat_tests_oot.csv`, `table_stat_tests_external.csv` |
| Bootstrap comparisons (run 0 and external) | `table_stat_tests_bootstrap_run0.csv`, `table_stat_tests_bootstrap_external.csv` |

Tables 1 and 12 are literature summaries and are not produced by the code.

### 5.2 Paper figures and the files that produce them

| Paper figure | File |
|---|---|
| Fig. 1 (architecture) | drawn manually, not generated |
| Fig. 2 (model comparison) | `fig2_model_comparison.png` |
| Fig. 3 (ablation ROC and PR) | `fig3_modality_ablation_roc_pr.png` |
| Fig. 4 (external ROC and PR) | `fig_external_modality_roc_pr.png` |
| Fig. 5 (reliability curves) | `fig5_calibration.png` |
| Fig. 6 (global SHAP importance) | `fig_shap_global.png` (data: `shap_global_importance.csv`, `shap_modality_share.csv`) |
| Fig. 7 (borrower-level SHAP) | `fig_shap_borrower_level.png` (data: `shap_borrower_level.csv`) |
| Fig. 8 (fusion weights by risk group) | `fig_fusion_weights.png` (data: `fusion_weights_by_risk_group.csv`, `fusion_weights_by_class.csv`, `fusion_weights_stability_across_runs.csv`) |

Additional figures not used in the paper: `fig4_all_models_roc_pr.png`, `fig6_cost_sensitive.png`, `fig_oot_modality_roc_pr.png`.

### 5.3 Other files

| File | Content |
|---|---|
| `tuning_log.csv` | All hyper-parameter configurations tried for every model and their validation PR-AUC |
| `run_config.json` | Settings, selected hyper-parameters, cohort audit, split sizes, software environment |
| `dataset_summary.csv` | Cohort construction counts, exclusions, prediction months |
| `sanity_checks.json` | Shuffled-label check (test AUC 0.471) |
| `run_log.txt` | Console log of the run |
| `results_all_runs.csv` | Per-run, per-model, per-protocol metrics (source of all summary tables) |
| `table_SUPPLEMENTARY_balanced_test_subsample.csv` | Metrics on a 1:1 test subsample (supplementary only; not representative of real prevalence) |
| `manuscript_snippets.md` | Text auto-generated from the run (cohort counts, configuration) |

---

## 6. Protocol details (as implemented)

**Cohort.** One loan = one record. t0 = first reporting month + 12 months (January to March 2008 for 2007Q1 loans). Features use only loan-month records with t at or before t0; the label uses only records after t0 (this separation is asserted in the code). Label = 1 if delinquency status reaches 3 or more, or the loan exits through a credit event (short sale, third-party sale, real-estate-owned, note sale), within 12 months after t0. Excluded: loans with no record at t0, loans already delinquent at t0, and loans that leave the portfolio within the window without an event (prepayment treated as censoring). Counts are in `dataset_summary.csv`.

**Modalities.** Financial (underwriting variables known at origination, 30 input columns after encoding) and behavioural/contextual (payment behaviour up to t0, loan age, paydown ratio, channel, seller, servicer, state; 79 input columns). In the code and figure legends this second modality is still labelled "alternative"; in the paper it is called the behavioural/contextual (loan-performance) modality. Both come from the same mortgage file.

**Random split.** Stratified 70/15/15 split at loan level, new split every run. Imputers, scalers and encoders are fitted on the training part only.

**Later-age snapshot (the `oot` protocol in the code).** Training and validation loans are taken at age 12 months; the test set is a random 30% of loans taken at age 24 months (test loans are disjoint from training loans and are redrawn in each run). This differs from a pure calendar-time split: the test loans have 24 months of history instead of 12, and training label windows for some loans extend past the prediction date of some test loans.

**External test.** Models trained on 2007Q1 (age 12 months) are tested on all eligible 2018Q1 loans (prediction dates January to March 2019). The 2018Q1 file is never used for training, tuning or thresholds.

**Tuning.** Six random configurations per model (the first is the default), selected on validation PR-AUC. Tuning is done once, on the run-0 split, and the selected configuration is used in all runs.

**Statistics.** The ten runs are repeated random splits of one cohort, so test sets overlap and the plain paired t-test and t-interval over runs are too optimistic. `corrected_statistics.py` applies the Nadeau-Bengio corrected resampled t-test (variance factor 1/k + n_test/n_train) and Holm adjustment over 12 comparisons. For the external test the test set is identical in all runs, so those intervals reflect training randomness only. Wilcoxon p-values in `table_stat_tests_across_runs*.csv` are uncorrected and descriptive.

**Note on the script header.** The docstring at the top of `mmai_mortgage_pipeline.py` mentions five runs and cost-sensitive weighting without resampling. That text is outdated. The configuration used for the paper is `RUNS = 10` and `TRAIN_BALANCE = "undersample"` (the positive-class weight is therefore 1.00, as shown in `run_log.txt`).

---

## 7. Reproducibility notes

- Random seeds are set for splits and initialisation (seed = run index). Multi-threaded and GPU training is not bit-wise deterministic, so repeated executions can differ slightly. All reported values come from the single execution stored in `results_v4/`.
- The pipeline does not store model predictions, so the bootstrap tables cannot be recomputed from the stored files without rerunning the pipeline. All other tables can be recomputed from `results_all_runs.csv`.

---

## 8. Citation and licence

If you use this code, please cite the paper and this archive: *[add DOI / repository link after upload]*.

*Add a licence file (for example MIT) before publishing the repository.*

---

## 9. Contact

Amit Kurchania, Marlabs LLC, Texas, USA: Amit.Kurchania@Marlabs.com
