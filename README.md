# The Blog Authorship Corpus
* NLP methods to predict gender, age & zodiac sign of the blogger
* Data preprocessing: remove non-alphabet chars, stop words, lemmatization
* Train different algorithms, choose best and save to the joblib format

## The Data
### Description
The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person. Each blogpost is presented as a separate row, with a blogger id# and the blogger’s self-provided gender, age, industry and astrological sign. (All are labeled for gender and age but for many, industry and/or sign is marked as unknown.)

Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label urllink.


## Prerequisites
Python version: 3.7      
You can set up virtual environment in one-step by using:
```
> conda conda env create -f environment.yml
```

## File Structure
    .
    ├── blobs                   # store trained model
    │   ├── __init__.py         # mark as package directory
    │   └── cls_model.joblib    # trained model in joblib format
    ├── data                    # store data
    │   ├── __init__.py         # mark as package directory
    │   └── blogtext.csv.zip    # blogtext data zipped
    ├── train                   # module contains training script
    │   ├── __init__.py         # mark as package directory
    │   └── train_model.py      # script load data, preprocess & train model
    ├── run.py                  # run train_model.py
    ├── bloggers.ipynb          # summary in jupyter notebook format
    └── README.md               # project summary, instructions


## Train Model
In your command line type:
```
> python run.py blogtext.csv.zip
```

## Blobs
Best trained model is stored in /blobs/cls_model.joblib
