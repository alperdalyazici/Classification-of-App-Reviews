# App Review Classification for Requirements Engineering

This repository serves as the base for my final year project at the University of Manchester, where I researched the classification of app reviews for requirements engineering using deep learning/large language models: CNN, XLNet and DistilBERT.

## Introduction

App reviews play a crucial role in understanding user requirements and preferences for app development. In this project, we focused on the classification of app reviews to support requirements engineering processes. By leveraging state-of-the-art deep learning models such as CNN, XLNet and DistilBERT, we aimed to automatically extract relevant information from app reviews, enabling developers to gain insights into user expectations and improve the software development lifecycle.

## Table of Contents
- [Dataset](#dataset)
- [Models](#models)
- [Experiments](#experiments)
- [Results](#results)


## Dataset

The dataset used in this project was obtained from two sources: the 'Augmented' and the 'Dataset Guzman Labelled'. Both datasets include reviews from various apps, such as Dropbox, Evernote, TripAdvisor, PicsArt, Pinterest, and WhatsApp. These reviews were classified into four classes: Feature Request (FR), User Experience (UE), Bug Report (BR), and Rating (RT).

The "Augmented" file has a total of 7311 reviews that are distributed into 1806 bug reports, 2117 user experiences, 1760 feature requests, and 1628 ratings. Similarly, the second file, "dataset_guzman_labelled," includes a total of 6159 reviews that are distributed into 990 bug reports, 2518 user experiences, 404 feature requests, and 1628 ratings

A Python script is written and used to randomly select 2000 reviews from each category across the two original datasets and create a new single dataset consisting of 8000 balanced reviews

## Models

The project utilizes the following deep learning models for app review classification:

- CNN: Convolutional Neural Network
- XLNet: Generalized Autoregressive Pretraining for Language Understanding
- DistilBERT: Distilled Bidirectional Encoder Representations from Transformers

Each model is implemented and provided with relevant training and evaluation scripts.

## Experiments

Two alternative techniques were applied to train and evaluate the deep learning models in this research project:

### Approach 1: Train-Test Split

In the first approach, the combined dataset was randomly split into a training set (50%), validation set (25%) and a testing set (25%). The models were trained on the training set using the Adam optimizer with a learning rate of 1e-5 for 5 epochs. A batch size of 8 was used, and a Sparse Categorical Crossentropy loss function was employed. 

### Approach 2: 5-Fold Cross Validation

For the second approach, 5-fold cross-validation was utilized to train and evaluate the models. The combined dataset was divided into 5 folds, ensuring each fold contained an equal proportion of samples from each class. The models were trained on each of the 5 folds using the same hyperparameters as in Approach 1. Performance measures such as accuracy, precision, recall, and F1-score were computed for each fold and averaged across all folds. To monitor the model's development and prevent overfitting, the performance on the validation set was used during training. 

## Results

The performance of the three models (CNN, XLNet and DistilBERT) was evaluated on four different evaluation metrics using two different experimental approaches.

### Approach 1: 50%-25%-25% Train-Test Split

- XLNet achieved the highest accuracy (84%) and F1-scores, as well as the highest precision scores on several tasks.
- CNN and DistilBERT had lower scores overall, but their performance varied across different tasks.
- The differences in architecture, training techniques, and the nature of tasks contributed to the variations in model performance.

### Approach 2: 10-Fold Cross Validation

- DistilBERT achieved the highest accuracy (92%) and F1-scores.
- All three models showed significant improvement in performance when evaluated using 5-fold cross-validation.
- The average F1-scores for each task and model increased, indicating better generalization capability.
- Cross-validation allowed for training on a larger amount of data, reducing the risk of overfitting and improving performance.

### Comparison of Models

- CNN was the most consistent model across different tasks, achieving balanced and impressive performance.
- XLNet and DistilBERT showed higher performance on specific tasks, such as Bug Report and User Experience, respectively.
- These results emphasize the importance of evaluating model performance on multiple tasks and using different evaluation approaches.

Overall, this research highlights the need for continued improvement and research in natural language processing models to better address complex software engineering tasks.
