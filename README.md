# ECG-lead-importance
Lead importance in 12-lead ECG
Project proposed for graduate level class: Machine Learning for Medical Application `ML4MA_project.pdf`

# Datasets
### Arythmia
Download *A large scale 12-lead electrocardiogram database for arrhythmia study* from [physionet](https://physionet.org/content/ecg-arrhythmia/1.0.0/).

### PTB-XL
Downloaded from [here](https://physionet.org/content/ptb-xl/1.0.3/).

# Replication
### Creating data files
Run `data_prep.ipynb` and `data_prep_ptb-xl.ipynb`.

### Training models
Run `train_model_mult_outputs.ipynb`. For single load models run `single_lead_mult_outputs.ipynb`.
For coalitions run `train_6_lead_model.ipynb`.

### Evaluation
Arrythmia dataset: `evaluate_internal_test.ipynb`.

PTB-XL: `eval_ptb-xl.ipynb`.

6 lead models: `eval_6_lead_model.ipynb`.

### Shap results
Run `shap_approximation_all.ipynb`.


