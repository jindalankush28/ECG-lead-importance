# ECG-lead-importance
Lead importance in 12-lead ECG

Project proposed for graduate level class: Machine Learning for Medical Application `ML4MA_project.pdf`

## Project Overview
This project investigates the relative importance of each lead in a 12-lead electrocardiogram (ECG) for diagnosing cardiac conditions. By optimizing lead selection, we aim to improve the efficiency and accessibility of ECG diagnostics, particularly in resource-constrained environments.

## Objectives
- Train models to predict cardiac conditions using a reduced number of ECG leads.
- Evaluate the diagnostic value of each lead using Shapley values.
- Validate models on external datasets to ensure generalizability.

---

## Motivation
Cardiovascular diseases (CVDs) are a leading cause of mortality worldwide. While 12-lead ECGs are valuable diagnostic tools, their complexity may limit accessibility. Reducing the number of required leads can:
- Lower costs
- Improve accessibility in underserved areas
- Enhance patient comfort
- Expedite diagnosis without compromising accuracy

---

## Significance
- **Cost-Effective Diagnosis:** Streamlines workflows and reduces financial burden.
- **Time Efficiency:** Accelerates clinical decision-making.
- **Accessibility:** Enables diagnostics in remote or resource-poor settings.
- **Personalized Care:** Facilitates tailored diagnostic approaches.
- **Advancing Knowledge:** Contributes to ECG analysis methodologies and clinical best practices.

---

### Shapley Value Approximation
- Monte Carlo sampling approach to estimate the contribution of each lead.
- Enables efficient evaluation without exhaustive computation of all lead combinations.

---


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

## References
1. Zheng, J., et al. *A large scale 12-lead electrocardiogram database for arrhythmia study*. PhysioNet (2022).
2. Wagner, P., et al. *PTB-XL: A large publicly available electrocardiography dataset*. Sci. Data (2020).
3. Attia, Z. I., et al. *Screening for cardiac contractile dysfunction using AI-enabled ECG*. Nat. Medicine (2019).
4. Štrumbelj, E., et al. *Explaining prediction models and individual predictions with feature contributions*. Knowl. Inf. Sys. (2014).

---

