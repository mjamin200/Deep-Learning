# Deep Learning Project - GAN-BERT for Text Classification

This repository contains code for training a GAN-BERT model for text classification on the SemEval-2024 dataset.

## Dataset
The dataset contains text samples from 6 classes: Human, ChatGPT, Cohere, Davinci, BloomZ, Dolly. The goal is to classify input text into one of these 6 classes.

The dataset is taken from Subtask B of SemEval-2024 Task 8. Only a small labeled portion of the data is used, the rest is treated as unlabeled.

## Models 
- A BERT classifier is first trained on the labeled data using HuggingFace pretrained BERT. An adapter module is used with BERT to reduce the number of trainable parameters [2].
- GAN-BERT model is then implemented following the architecture in [1]. This contains:
  - Discriminator (D): A feedforward network that classifies vector representations from BERT into K+1 classes (K real classes + 1 fake class)
  - Generator (G): Made up of G1 which maps noise to vector representations and G2 which is a pretrained BERT model independent of D's BERT.
  
G1 and G2 are trained adversarially along with D.

## Files
The repository contains the following files:

- `BERT.ipynb`: Notebook for training BERT classifier
- `BERT_Adapter.ipynb`: Notebook for training BERT classifier with adapter 
- `G1.ipynb`: Notebook for training generator G1
- `G2.ipynb`: Notebook for training generator G2
- `utility.py`: Utility functions for BERT classifier
- `utilityG.py`: Utility functions for GAN-BERT

## Acknowledgements
We refer to the paper 'GAN-BERT for Automated Essay Scoring' by Griffin Holt and Theodore Kanell from Stanford University (CS224N course) for implementing the generator networks G1 and G2.

The complete reference is:

Holt, Griffin and Kanell, Theodore. "GAN-BERT for Automated Essay Scoring." Stanford CS224N Custom Project, Department of Electrical Engineering and Department of Computer Science, Stanford University, 2022.

## Contact 

Please reach out with any questions!

Email: m.j.amin200@gmail.com

## References
[1] Croce, Danilo, Giovanni Castellucci, and Roberto Basili. "GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020. 

[2] Houlsby, Neil, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. "Parameter-efficient transfer learning for NLP." International Conference on Machine Learning. PMLR, 2019.
