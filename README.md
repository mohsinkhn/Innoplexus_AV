## Innoplexus_AV
Repository with 1st place solution for Innoplexus Hackathon hosted on Analytics Vidhya.

To get submission that scored 1st on private and 2nd on public leaderboards, just make appropriate changes to config.py file and run `bash run_all.sh`   

### Dependencies
- beautifulsoup4 >= 4.6.0
To install `pip install beautifulsoup4`

### 
### Approach
* Validation
- The idea here was to mimic train and test split. As per the problem statement combination of domain and tag could only be present in either train or test. I made a new column adding string of domain and tag. I use this new column to generate cross validation folds using GroupKFold

* Modelling
  I had two very simple models.
  * **Model 1 - TFIDF + Logisticregression over Webpage data**
    - I generated tfidf features over both word and character ngrams of parsed html webpage data. I ran Logistic regression over these tfidf features.
  * **Model 2 - TFIDF + Logreg over URL's + Model 1**
    - I generated tfidf features over both word and character ngrams of url text. I also concatenate out-of-fold features generated from previous model here and then just run Logistic regression over all features.

* Post-processing
  - To utlize the train/test split strategy, wherein any domain+tag combination cannot appear in test. While, making prediction for test, I remove tag present in train for a given domain, and pick the one with maximum probability amongst all others.
