# Word Classifier

The present project is a classifier system that takes in a string and output one of the following
classes - date, location, random string, company name, physical goods,
other.

### Requirements

1) Python 3.7
2) The embeddings repository: GoogleNews-vectors-negative300.bin.gz. Please Download it from:
 ```https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz```
 and save it under data/word_embeddings/
3) The required python libraries are located under requirements.txt. In order to install it please type
```pip install -r requirements.txt```

### Train the classifier
If you would like to use the already pre-trained models, please skip this step. 
The models are stored under data/pre_trained_models/
```
cd src/
python -m Trainer
```


### Use the Classifier

```
cd src/
python -m Classifier
```


