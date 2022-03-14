# Arabic Dialect Detection
This project experiments with detecting Arabic dialects using a simple ML approach (Naive Bayes) and a RNN model.

## Model details
#### Complement Naive Bayes
The first model used is a Complement Naive Bayes. I chose it because the data I had access to was fairly unbalanced and because using SVM with cross validation would take too long to train. The model achieves an average accuracy of 54% (highest f1 score: 0.69, lowest f1 score:0.25)

#### RNN
Here I used a NN comprising of an embedding layer with dimensions of 300 followed by 2 LSTM layers and a dropout layer to avoid overfitting. This model suffers from a lack of sufficient data as well as suboptimal word representations. The model achieves an average accuracy of 48% and a more severe difference in performance between classes (highest f1 score: 0.75, lowest f1 score:0.06)

## Deploying the models

Clone the repo 
```bash
  git clone https://github.com/hattoum/arabic_dialect_detection
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

## Usage
Once the model is deployed, it expects GET requests to ```localhost:8000/predict``` and the parameter ```text``` for the text (string) you need to predict the dialect of.
