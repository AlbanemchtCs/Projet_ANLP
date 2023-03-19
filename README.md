# :globe_with_meridians: Intents Classification for Neural Text Generation Project
Project for the Advanced Natural Language Processing course at CentraleSupélec.

The subject of the project can be found on [GitHub](https://github.com/PierreColombo/NLP_CS/blob/main/project/project_3_intent.md).

## 🎯 Objective
The aim of the project is to implement an intent classifier.

## 📌 Context and issues
The identification of both Dialog Acts (DA) and Emotion/Sentiment (E/S) in spoken language is an important step toward improving model performances on spontaneous dialogue task. Especially, it is essential to avoid the generic response problem, i.e., having an automatic dialog system generate an unspecific response — that can be an answer to a very large number of user utterances. DAs and emotions are identified through sequence labeling systems that are trained in a supervised manner DAs and emotions have been particularly useful for training ChatGPT.

## :page_facing_up: Problem statement
We start by formally defining the Sequence Labelling Problem. At the highest level, we have a set $D$ of conversations composed of utterances, i.e., $D = (C_1,C_2,\dots,C_{|D|})$ with $Y= (Y_1,Y_2,\dots,Y_{|D|})$ being the corresponding set of labels (e.g., DA,E/S). At a lower level each conversation $C_i$ is composed of utterances $u$, i.e $C_i= (u_1,u_2,\dots,u_{|C_i|})$ with $Y_i = (y_1, y_2, \dots, y_{|C_i|})$ being the corresponding sequence of labels: each $u_i$ is associated with a unique label $y_i$. At the lowest level, each utterance $u_i$ can be seen as a sequence of words, i.e $u_i = (\omega^i_1, \omega^i_2, \dots, \omega^i_{|u_i|})$.

The goal is to predict Y from D !

## 🤔 Technical choices
### 📊 Dataset
We have taken the dataset used in the following original paper [Code-switched inspired losses for generic spoken dialog representations](https://arxiv.org/pdf/2108.12465.pdf). The full dataset is available on [Hugging Face](https://huggingface.co/datasets/miam).
Here is the composition of the different sub-datasets:

| Dataset name          | Language                                             | Train                    | Valid                    | Test                    |
|--------------------------|----------------------------------------------------|--------------------------|--------------------------|-------------------------|
| dihana                   | Spanish                                           | 19063                    | 2123                     |2361                     |     
| ilisten                  | Italian                                             | 1986                     | 230                      |971                      |    
| loria                    | French                                           | 8465                     | 942                      |1047                     |    
| maptask                  | English                                            | 25382                    | 5221                     |5335           |             
| vm2                      | German                                           | 25060                    | 2860                     |2855   |         

We decided to implement our model on the different datasets in order to see if the multilingual model that we implemented works well.

### 🔡 Tokenizer 
We implement an mBERT tokenizer that works by splitting words into subwords using the subword tokenization algorithm, adding special tokens and encoding the tokens into embeddings. This approach allows the mBERT model to handle multiple languages and to generalise better by solving the out-of-vocabulary problem.

### 🤖 Model
We choose to use the mBERT (multilingual BERT) model which is a multilingual version of the BERT (Bidirectional Encoder Representations from Transformers), which is pre-trained on a large corpus of texts in several languages.

mBERT allows for processing multiple languages without the need for language-specific models for each language, which enables better generalization.

Moreover, mBERT is particularly efficient for processing complex texts, such as scientific or technical texts, legal texts or government documents.

## :card_index_dividers: Segmentation
Our directory is split into two python files, two jupyter notebooks, two markdown files, a .gitinore file and a text file for the requirements :

```bash 
.
├── README.md
├── CONTRIBUTING.md
├── .gitignore
├── requirements.txt 
├── miam.py
├── datasets_visualization.ipynb
├── train_test.py
└── train_test.ipynb

```

- ``README.md`` contains all the information about the project in order to install it.
- ``CONTRIBUTING.md`` contains all the information on standards and practices for collaboration and project management.
- ``.gitignore`` contains files that should be ignored when adding files to the Git repository.
- ``requirements.txt`` contains a list of Python modules and libraries that need to be installed, and their specific version.
- ``miam.py`` is the python file that extracts the datasets.
- ``datasets_visualization.ipynb`` enables to visualize the different datasets with their class.
- ``train_test.py`` is the python file that allows us to run our m-BERT model with the tokenizer.
- ``train_test.ipynb`` is based upon the python file ``train_test.py`` and allows to save our results for the different datasets.

## :wrench: Installation
To run the code, we recommend on a terminal only:

1. First of all, make sure you have installed a `python` version higher than 3.9. We recommend a conda environment with the following command :
```bash
conda create --name intent_classification python=3.9
```
- To activate the environment :
```bash
conda activate intent_classification
```
- To access the directory : 
```bash
cd projet_anlp
```

2. You must then install all the `requirements` using the following command :
```bash
pip install -r requirements.txt
```

3. Run the model using the following command :
```bash
python3 train_test.py
```

## :pencil2: Authors
- MICHOT Albane
- NONCLERCQ Rodolphe



