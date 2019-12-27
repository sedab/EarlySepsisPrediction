# Data Preprocessing
Strategies to impute missing values <br>

* Baseline (`missing_vals_baseline.ipynb`): interpolation for vital signal data; fill zeros for lab test data;
drop *Unit1*, *Unit2*, *EtCO2* features
* LDS (`missing_vals_LDS.ipynb`): interpolation for vital signal data;
run LDS twice on each patient to fill missing lab test data; drop *Unit1*, *Unit2*, *EtCO2* features
