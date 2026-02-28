# explainable-AI-with-EMR

## Description
Explainable AI Illuminates Tissue Pathology and Psychosocial Drivers in Opioid Prescription for Low Back Pain. Predicting opioid prescription patterns for non-specific chronic lower back pain (ns-cLBP) requires robust patient profiling and classification strategies, amidst conservative management strategies. LBP patient profiles were constructed from clinical chart (demographics, social determinants, diagnoses, medications) and radiology report (MRI-confirmed diagnoses) electronic medical record data. One-vs-one (OVO) and one-vs-rest (OVR) classification frameworks systematically compared bi-directional and class-wise decisions across no, NSAID, and opioid medication. 

## Getting Started

### Dependencies
The scripts in this repository have been tested with Python .


### Installing
1. Set up a github API Key
2. Clone the repo
```
git clone git@github.com:michelle-tong18/explainable-AI-with-EMR.git
```

## Pipeline

### Extract electronic medical records (EMR). 
1. Specific inclusion and exclusion criteria to identify a patient cohort. 
2. (Optional) Extract MR images for patient cohort.
3. Extract tabular medical chart and clinical note data for patient cohort.

### Prepare patient profiles of clinical-relevant features.
1. Clean the EMR data for each table extracted from the EMR relational database, and specify values for every clinically meaningful category. For example, review of the primary insurance category resulted in the categorical variables: PPO, Medicare, Medicaid, HMO/POS/EPO, and others.
1. Extract the presence of pathology from radiology reports using custom prompts and GPT 4.0.
2. Impute missing demographic information by evaluating missingness type (MCAR, MAR), artifical missingness imputation, and imputation affect on variable distributions.
3. Consolidate patient profiles by joining multiple EMR tables and map categorical variables to numerical values.

### Model clinical decision making.
0. Compare pathology reporting in medical charts and image-confirmed radiology reports.
1. Train tree-based classification models (Random Forest, Bagging, AdaBoost, and XGBoost) using 5-fold cross-validation with hyperparameter tuning. Implement both one-vs-one and one-vs-rest classification strategies to predict prescription categories: no medication, NSAIDs, or opioids. Select the optimal models based on validation performance, then evaluate model generalization on the test set. Finally, analyze SHAP feature importance to identify key clinical predictors associated with prescription patterns.
2. Generate figures for model and feature interpretation.

## License
Distributed under the MIT License. See LICENSE.txt for more information.

## Citations
* Tong MW, Ziegeler K, Kreutzinger V, Majumdar S. Explainable AI reveals tissue pathology and psychosocial drivers of opioid prescription for non-specific chronic low back pain. Sci Rep 15, 30690 (2025).  .doi: [https://doi.org/10.1038/s41598-025-13619-7](10.1038/s41598-025-13619-7)