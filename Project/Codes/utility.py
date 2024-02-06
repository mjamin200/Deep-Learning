import os
import gdown
import json
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
from datasets import load_dataset
from collections import Counter
from transformers import TrainingArguments, Trainer
 
    

class DataHandler:
    def __init__(self):

        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def get_path(self, folder_name):
        """
        Get the absolute path of a specified folder within the parent directory.

        Parameters:
        - folder_name (str): Name of the folder.

        Returns:
        - str: Absolute path of the specified folder within the parent directory.
        """        
        folder_path = os.path.join(self.project_folder, folder_name)

        # Create folder
        os.makedirs(folder_path, exist_ok=True)

        return folder_path

    def download_file_from_drive(self, file_id, file_name):
        """
        Download a file from Google Drive using its file ID.

        Parameters:
        - file_id (str): The ID of the file on Google Drive.
        - output_file_path (str): The local path where the file will be saved.

        Returns:
        - None
        """
        path = os.path.join(self.get_path('Dataset'), file_name)
        url = f'https://drive.google.com/uc?id={file_id}'

        gdown.download(url, path, quiet=True)

    def read_jsonl_file(self, file_name):
        """
        Read a JSON Lines file and load it into a dataset using the Hugging Face datasets library.

        Parameters:
        - file_name (str): The name of the JSON Lines file to be read.

        Returns:
        - dataset: The dataset loaded from the JSON Lines file.
        """
        file_path = os.path.join(self.get_path('Dataset'), file_name)
        self.dataset = load_dataset("json", data_files=file_path)

        return self.dataset

    def download_and_process_datasets(self, Download = True):
        """
        Download and process training and dev datasets.

        Parameters:
        - Download (bool): Flag indicating whether to download the datasets.

        Returns:
        - None
        """
        # file IDs and file names for both train and dev
        file_id_train = '1k5LMwmYF7PF-BzYQNE2ULBae79nbM268'
        file_name_train = 'subtaskB_train.jsonl'

        file_id_dev = '1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE'
        file_name_dev = 'subtaskB_dev.jsonl'

        train_output_path = os.path.join(self.get_path('Dataset'), file_name_train)
        dev_output_path = os.path.join(self.get_path('Dataset'), file_name_dev)

        if Download :
           # Download dataset
            self.download_file_from_drive(file_id_train, train_output_path)
            self.download_file_from_drive(file_id_dev, dev_output_path)

        # Read file
        self.dataset_train = self.read_jsonl_file(train_output_path)
        self.dataset_dev = self.read_jsonl_file(dev_output_path)

    
    def tokenize_datasets(self, tokenizer):
        """
        Tokenize the training and dev datasets using the provided tokenizer.

        Parameters:
        - tokenizer: The tokenizer to be used for tokenization.

        Returns:
        - ds_train: Tokenized training dataset.
        - ds_val: Tokenized dev dataset.
        """
        ds_train = self.dataset_train.map(lambda example: tokenizer(example['text'], max_length=64, truncation=True), batched=False)
        ds_val = self.dataset_dev.map(lambda example: tokenizer(example['text'], max_length=64, truncation=True), batched=False)

        ds_train = ds_train.remove_columns(['text', 'source', 'id', 'model'])
        ds_train = ds_train.rename_column('label', 'labels')

        ds_val = ds_val.remove_columns(['text', 'source', 'id', 'model'])
        ds_val = ds_val.rename_column('label', 'labels')

        return ds_train['train'], ds_val['train']
    
    def distribution_plot(self, train_dataset_dict, val_dataset_dict, id2label, output_filename='distribution_plot.png'):
        """
        Create distribution plots for the samples in each label category for both training and validation datasets.
        Save the plots to the plot folder.

        Parameters:
        - train_dataset_dict: Training DatasetDict object.
        - val_dataset_dict: Validation DatasetDict object.
        - id2label: Dictionary mapping label indices to label names.
        - output_filename: Filename for the saved plot (default is 'distribution_plot.png').
        """

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

        # Plot for Training Dataset
        train_labels = train_dataset_dict['labels']
        train_labels = [id2label[label] for label in train_labels]
        train_label_counts = Counter(train_labels)

        train_color_palette = sns.color_palette("husl", n_colors=len(train_label_counts))

        sns.countplot(x=train_labels, palette=train_color_palette, hue=train_labels, ax=axes[0], legend=False)
        axes[0].set_xlabel('Labels')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title(f'Distribution of Samples for Each Label in Training Dataset')

        # Plot for Validation Dataset
        val_labels = val_dataset_dict['labels']
        val_labels = [id2label[label] for label in val_labels]
        val_label_counts = Counter(val_labels)

        val_color_palette = sns.color_palette("husl", n_colors=len(val_label_counts))

        sns.countplot(x=val_labels, palette=val_color_palette, hue=val_labels, ax=axes[1], legend=False)
        axes[1].set_xlabel('Labels')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title(f'Distribution of Samples for Each Label in Validation Dataset')

        # Save the plot
        output_file_path = os.path.join(self.get_path('Plots'), output_filename)
        plt.savefig(output_file_path)

        plt.show()

        # Print label counts
        print("\nLabel Counts for Training Dataset:")
        for label, count in train_label_counts.items():
            print(f"{label}: {count}")

        print("\nLabel Counts for Validation Dataset:")
        for label, count in val_label_counts.items():
            print(f"{label}: {count}")

        print("\n\n")

def compute_metrics(p, metric):
    """
    Compute evaluation metrics for the given predictions and labels using the specified metric.

    Args:
        p (tuple): A tuple containing predictions and labels.
        metric: The metric to be used for evaluation.

    Returns:
        dict: A dictionary containing the computed accuracy metric.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    results = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": results["accuracy"]}

def train_model(model, train_set, val_set, epochs, batch_size, tokenizer, data_collator, metric, device, seed_value=42, use_reentrant=False):
    """
    Train the provided model using the given training and validation datasets.

    Args:
        model: The model to be trained.
        train_set: Training dataset.
        val_set: Validation dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        tokenizer: Tokenizer for processing input data.
        data_collator: Data collator for batch processing.
        metric: Evaluation metric.
        device: Device to use for training (e.g., 'cuda' for GPU or 'cpu').
        seed_value (int): Seed value for reproducibility.
        use_reentrant (bool): Whether to use reentrant checkpointing. Default is False.

    Returns:
        list: List of accuracies at each epoch during training.
    """

    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="ouputFineTuning",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=1e+10,
        report_to='tensorboard',
    )
    model = model.to(device)

    # Initialize Trainer with use_reentrant parameter
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, metric),
    )

    # Train the model
    history = trainer.train()

    training_logs = trainer.state.log_history
    accuracies = [training_logs[i]["eval_accuracy"] for i in range(len(training_logs) - 1)]

    return accuracies




def plot_accuracy_vs_percentage(percentages, accuracies, label=None):
    """
    Generate a line plot to visualize the relationship between annotated percentages and corresponding accuracies.

    Parameters:
    - percentages (list): List of percentages for annotated data.
    - accuracies (list): List of corresponding accuracies.
    - label (str): Label for the legend (default is None).

    Returns:
    - None: Displays the generated plot.
    """
    plt.semilogx(percentages, accuracies, marker='o', color='black', label=label)
    plt.xticks(percentages, [f"{p}%" for p in percentages])

    plt.xlabel('Annotated')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    plt.savefig('../Plots/bert_accuracies.png')

    






