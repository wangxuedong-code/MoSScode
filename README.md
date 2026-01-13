# Medical Classification Problem Solving Framework

## Project Overview
Electronic Health Records (EHRs) encompass patient-related information and have significantly contributed to the development of predictive models in the healthcare field. However, inevitable missing values in EHRs pose challenges to exploring underlying patterns, limiting the performance and effectiveness of most machine learning methods. In this paper, we focus on EHRs with missing values and propose the Mixture of Subclassifiers with complete Submatrix (MoSS) framework to tolerate missing values, instead of conducting deletion or imputation on the raw data.

## Supplementary Note
Details about the datasets used in this study are as follows:
- The 4 UCI datasets are available for direct use. 
- To obtain the CECMed dataset, researchers are required to contact the corresponding author Tang Wen to apply for data usage permission and comply with the relevant data sharing agreements.

## Environment Setup
Configure the running environment by installing dependencies according to the requirements file:

```bash
# Install required packages
pip install -r requirements.txt

# Run the model training script
python model_trainer.py
