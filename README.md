# DSGAT
Here we propose a novel end-to-end approach named DSGAT for drug-side effect frequency prediction by using graph 
attention networks (GAT) . DSGAT learns the drug representation from the molecular graph, and has the ability to 
predict the frequencies of side effects for cold start drugs that do not appear in the training data.
# Requirements
* python == 3.8.5
* pytorch == 1.6
* Numpy == 1.21.0
* scikit-learn == 0.23.2
* scipy == 1.5.0
* rdkit == 2020.03.6
* matplotlib == 3.3.1
* networkx == 2.5

# Files:
1.original_data

This folder contains our original side effects and drugs data.

* **Supplementary Data 1.txt:**     
  The standardised drug side effect frequency classes used in our study. 


* **Supplementary Data 2.txt:**  
The postmarketing drug side effect associations from SIDER and OFFSIDES.
  

* **Supplementary Data 3.txt:**  
Main or High‐Level Term (HLT) Medical Dictionary for Regulatory Activities (MedDRA) terminology classification for each side effect term.
  

* **Supplementary Data 4.txt:**  
High‐Level Group Term Medical Dictionary for Regulatory Activities (MedDRA) terminology classification for each side effect term.
  


2.data_WS/data_ICS/data_750+9
* **mask_mat.mat**   
The file is a mark that randomly divides all positive samples into 10 and generates training sets and test sets for cross validation. If this file is missing, it will be randomly generated again when the code runs.


* **robustness_test.mat**     
The test set samples randomly selected in the robustness test, in which the number of test sets is 10% of all drugs. If this file is missing, the code is regenerated at run time.


* **raw_frequency_750.mat**  
The original frequency matrix of side effects of 750 drugs, including frequency matrix 'R', drug name 'drugs' and side effect name' sideeffect '.


* **frequency_750+9.mat**      
The original frequency matrix of side effects of drugs in 750 drugs and 9 independent test sets, and the position of 9 drugs is before 750 drugs.
  

* **side_effect_label_750.mat**  
Label vector for 994 side effects


* **drug_SMILES_750.csv**   
SMILES of 750 drugs.


* **drug_SMILES_759.csv**   
SMILES files for 750 initial drugs and 9 independent test sets, and the position of 9 drugs is before 750 drugs.

# Code 

WS_v4.py: Warm start test of 750 drugs.

ICS.py: Cold start test of 750 drugs

ics_750+9.py: Cold start testing of drugs in 9 independent test sets.

Robustness.py: Robustness test of cold start prediction.

Net.py: It defines the model used by the code.

smiles2vector.py: It defines a method to calculate the smiles of drugs as vertices and edges of a graph.


utils.py: It defines some other functions, such as performance indicators, dataset classes, and so on.


# Run


model: Define the model used.

epoch: Define the number of epochs.

lr: Define the learning rate.

lamb: Define weights for unknown associations.

eps: Define the difference between the forecast score and the label.

train_batch: Define the number of batch size for training and testing.

tenfold: Use ten fold cross validation

Example:
```bash
python WS_v4.py --tenfold --save_model --epoch 3000 --lr 0.0001
```

# Contact
If you have any questions or suggestions with the code, please let us know. Contact Xianyu Xu at xxy45@tust.edu.cn