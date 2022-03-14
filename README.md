# Arabic Dialect Detection
This project experiments with detecting Arabic dialects using a simple ML approach (Naive Bayes) and a RNN model.

## Deploying the models

Clone the repo 
```bash
  git clone https://link-to-project
```
Install requirements 
```bash
pip install -r requirements.txt
```
<br/>
Fetch the data by running ```/fetching/fetch.ipynb``` 
<br/><br/>


Preprocess the text ```/preprocessing/preprocessing.ipynb```
<br/><br/>

Train both models in ```/training```
<br/><br/>

Start the server from ```/deploying```
```bash
uvicorn main:app --reload
```
