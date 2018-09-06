## Innoplexus_AV
Repository with 2nd place solution for Innoplexus Hackathon hosted on Analytics Vidhya.

To get submission that scored 2nd on both private and public leaderboards, just make appropriate changes to config.py file and run `bash run_all.sh`   

### Dependencies
- beautifulsoup4 >= 4.6.0
To install `pip install beautifulsoup4`

### 
### Approach
* Validation

* Modelling
  I had two very simple models.
  * **Model 1 - TFIDF + Logisticregression over Webpage data**
  * **Model 2 - TFIDF + Logreg over URL's + Model 1**

* Post-processing
