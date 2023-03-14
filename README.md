# :globe_with_meridians: Projet Intents Classification for Neural Text Generation
Projet pour le cours d'Advanced Natural Language Processing à CentraleSupélec. 

Le sujet du projet peut être récupéré sur [GitHub]{https://github.com/PierreColombo/NLP_CS/blob/main/project/project_3_intent.md}.

## 🎯 Objectif
Le but du projet est d'implémenter un classificateur d'intention. 

## 📌 Contexte et enjeux
L'identification à la fois des actes de dialogue (DA) et des émotions/sentiments (E/S) dans le langage parlé est une étape importante pour améliorer les performances des modèles sur les tâches de dialogue spontané. En particulier, il est essentiel d'éviter le problème de la réponse générique, c'est-à-dire qu'un système de dialogue automatique génère une réponse non spécifique - qui peut être une réponse à un très grand nombre d'énoncés de l'utilisateur. Les DAs et les émotions sont identifiés grâce à des systèmes d'étiquetage de séquences qui sont formés de manière supervisée. Les DAs et les émotions ont été particulièrement utiles pour former ChatGPT.

## :page_facing_up: Énoncé du problème
Nous commençons par définir formellement le problème d'étiquetage de séquences. Au niveau le plus élevé, nous avons un ensemble de conversations composées d'énoncés $D$, c'est-à-dire que $D = (C_1,C_2,\dots,C_{|D|})$ avec $Y= (Y_1,Y_2,\dots,Y_{|D|})$Y est l'ensemble correspondant d'étiquettes (par exemple, DA, E/S). À un niveau inférieur, chaque conversation $C_i$ est composée d'énoncés $u$, c'est-à-dire que $C_i= (u_1,u_2,\dots,u_{|C_i|})$ avec $Y_i = (y_1, y_2, \dots, y_{|C_i|})$ étant la séquence d'étiquettes correspondante : chaque $u_i$ est associé à une étiquette unique $y_i$. Au niveau le plus bas, chaque énoncé $u_i$ peut être vu comme une séquence de mots, c'est-à-dire, $u_i = (\omega^i_1, \omega^i_2, \dots, \omega^i_{|u_i|})$.

## 🤔 Choix techniques
### 📊 Dataset
Nous avons repris le dataset utilisé dans le papier original suivant [Code-switched inspired losses for spoken dialog representations]{https://aclanthology.org/2021.emnlp-main.656/}. Le dataset complet est disponible sur [Hugging Face]{https://huggingface.co/datasets/miam}.
Voici la composition des différents sous-datasets:

| Nom du dataset           | Langue                                             | Train                    | Valid                    | Test                    |
|--------------------------|----------------------------------------------------|--------------------------|--------------------------|-------------------------|
| dihana                   | Espagnol                                           | 19063                    | 2123                     |2361                     |     
| ilisten                  | Italie                                             | 1986                     | 230                      |971                      |    
| loria                    | Français                                           | 8465                     | 942                      |1047                     |    
| maptask                  | Anglais                                            | 25382                    | 5221                     |5335           |             
| vm2                      | Allemand                                           | 25060                    | 2860                     |2855   |         

Nous choisissons le sous-dataset `loria` qui semble un bon compromis entre nombre d'énoncés et précision des résultats.

### 🔡 Tokenizer
Nous faisons le choix d'utiliser un tokenizer mBERT (multilingual BERT) qui est une version multilingue du modèle BERT (Bidirectional Encoder Representations from Transformers), qui est pré-entraîné sur un grand corpus de textes dans plusieurs langues.

mBERT permet de traiter plusieurs langues sans avoir besoin de modèles de langage spécifiques pour chaque langue, ce qui permet une meilleure généralisation.

De plus, mBERT est particulièrement efficace pour le traitement de textes complexes, tels que les textes scientifiques ou techniques, les textes juridiques ou les documents gouvernementaux.

### 🤖 Modèle



## :card_index_dividers: Segmentation
Notre répertoire est segmenté en X fichiers python, X jupyter notebooks, deux fichiers markdown, un fichier .gitinore et un fichier texte pour les requirements :

```bash 
.
├── README.md
├── CONTRIBUTING.md
├── .gitignore
├── requirements.txt 
├── X
│     ├── X
│     └── X
└── X
      └──  x

```

- ``README.md`` contient l'ensemble des informations sur le projet pour pouvoir l'installer.
- ``CONTRIBUTING.md`` contient l'ensemble des informations sur les normes et les pratiques de collaboration et de gestion du projet.
- ``.gitignore`` contient les fichiers qui doivent être ignorés lors de l'ajout de fichiers au dépôt Git.
- ``requirements.txt`` contient la liste des modules et des bibliothèques Python qui doivent être installés, ainsi que leur version spécifique.

## :wrench: Installation
Pour lancer, nous vous recommandons sur un terminal uniquement :

1. Tout d'abord, assurez-vous que vous avez installé une version `python` supérieure à 3.9. Nous vous conseillons un environnement conda avec la commande suivante : 
```bash
conda create --name intent_classification python=3.9
```
- Pour activer l'environnement :
```bash
conda activate intent_classification
```
- Pour accéder au répertoire : 
```bash
cd projet_anlp
```

2. Vous devez ensuite installer tous les `requirements` en utilisant la commande suivante :
```bash
pip install -r requirements.txt
```

Exécuter ensuite les notebooks jupyter dans l'ordre suivant : 

1. X
2. X
3. X
4. X

## :pencil2: Auteurs
- MICHOT Albane
- NONCLERCQ Rodolphe



