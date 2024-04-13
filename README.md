# Project Name

In this project, we will build a Question-answering system using the Cross-Encoder model as the information retrieval, pre-train Bidirectional Encoder Representation from Transformer (BERT) using the SQuAD dataset as the reader, and GPT 3.5 as the generator. 
Firstly, for each question, the system will retrieve relevant articles using a retrieval model. The retrieval model will return the top 2 articles. Then, the reader model will extract the answer from the articles. Lastly, the answer that has a probability greater than 0.5 will be generated using a generative model.


This project will use Streamlit as a web-based app.

![nlp_pipeline-pipeline drawio](https://github.com/robinviltoriano/NLP_Assignment_2/assets/153576708/22bbac24-3dcb-4eb3-ac2b-c8996e75992e)


## Setup
To run this project: 
```
$ pip install -r requirements.txt
$ streamlit run app.py
```

## Contributors:
* Nguyen Dang Thinh
* Robin Viltoriano
* Pham Ky Cuong



