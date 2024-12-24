1. Lipinski Properties of metabolites: Lipinski_Properties.csv
  The file contains values for hydrogen bond donors, acceptors, molecular weights, logP and druglikeness(QED) values of all the metabolites in our dataset
2. Gram stain and pathogenicity of 312 bacterial species: BacteriaGramStainPathogenecity.tsv
3. Chemical categories of the compounds: compounds_categories.csv
4. Number of proteins, enzymes binding to metabolite per bacterial species: ProteinEnzymesperTaxon.csv
5. Example data for BRENDA substrate model: ClassificationSubstrateBrendaExample.csv
6. Example data for BRENDA product model: ClassificationProductBrendaExample.csv
   `Instances in substrate and product models contain Pubchem ID, Enzyme ID embedded into 1024 dimensional vector`
7. Example data for consumption model: Classification_5_Enzymes.csv
8. Example data for production model: Classification_5_Enzymes_product.csv
   `Instances in substrate and product models contain Pubchem ID, list of enzymes binding metablites sorted by STITCH scores`
    `Enzymes in the list are encoded as 1024 dimensional vector calculated using EC2Vec and metabolites encoded as 300 mol2vec vector`

Link: EC2Vec (https://github.com/MengLiu90/EC2Vec)
