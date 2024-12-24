# Gut Microbe Metabolite Interaction

The repository provides data driven approach for generating negative dataset and random forest binary classification models for substrate, product, consumption and production predictions

Four classification models are provided in this repository:

1. Brenda Substrate Prediction
2. Brenda Product Prediction
3. Metabolite consumption by gut microbes
4. Metabolite production by gut microbes

## Dependencies
1. pandas 1.5.0
2. sklearn 1.0.2
3. matplotlib 3.3.2

## Input data to the models
The EC numbers were embedded using EC2Vec (https://github.com/MengLiu90/EC2Vec). <br>
The metabolites were embedded using mol2vec (https://github.com/samoturk/mol2vec). <br>

The above embeddings were concatenated together as the input features to the classifiers.
1. Instances for the BRENDA substrate and product models contain chemical (encoded as 300 vector mol2vec) and EC number (encoded as 1024 vector using EC2Vec)
2. Instances for the consumption and production models contain chemical (encoded as 300 vector mol2vec) and list of enzyme ids (each enzyme encoded as 1024 vector using EC2Vec).<br>
   The `./Data` contains the exmaples of the input data for BRENDA substrate and product models and consumption and production models.

## Model training and usage
To execute the classifiers, prepare the data as per the provided instructions and place it within the ./Data/ directory.
1. The python script `RF_BREDNA_classifier_5_fold_cv.py` implements the BRENDA substrate or product classifer using 5-fold cross-validation
2. The python scripts `RF_consumption_classifier.py` and `RF_production_classifier.py` implement respective classifiers using the train/test split protocol.
3. The python scripts `RF_consumption_classifier_5_fold_cv.py` and `RF_production_classifier_5_fold_cv.py` implement respective classifiers using the 5-fold cross-validation protocol.
