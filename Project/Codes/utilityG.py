import os
import gdown
import json
import torch
import logging
import seaborn as sns
import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, random_split
from transformers import BertForSequenceClassification, set_seed

Seed = 42
random.seed(Seed)
np.random.seed(Seed)
torch.manual_seed(Seed)
set_seed(Seed)


class GanBertDataset(Dataset):
    def __init__(self, Percentage, Dataset_Name, Seed=42, Download_Dataset=True):
        """
        Initialize a GAN-BERT dataset.

        Parameters:
        - Percentage (float): The percentage of the dataset to use.
        - Dataset_Name (str): Name of the dataset ('Train' or 'Dev').
        - Seed (int): Random seed.
        - Download_Dataset (bool): Flag to download the dataset from Google Drive.

        Returns:
        - None
        """
        
        # Download the dataset from Google Drive
        self.download_file_from_drive(Dataset_Name, Download=Download_Dataset)

        # Read the JSON Lines file and load it into the dataset
        self.dataset = self.read_jsonl_file(file_name=self.path, Dataset_Name=Dataset_Name, Percentage=Percentage, Seed=42)

        torch.manual_seed(Seed)
        random.seed(Seed)
        
        size = len(self.dataset)
        resize = int(size * 1)
        # Use random_split to create a subset of the dataset
        self.dataset = random_split(self.dataset, [size - resize, resize], generator=torch.Generator().manual_seed(Seed))[1]
        
        
 

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


    def download_file_from_drive(self, Name_Dataset, Download=True):
        """
        Download dataset files from Google Drive.

        Parameters:
        - Name_Dataset (str): Name of the dataset ('Train' or 'Dev').
        - Download (bool): Flag to control whether to download the dataset.

        Returns:
        - None
        """
        if Name_Dataset == 'Train':
            file_name = 'subtaskB_train.jsonl'
            url = 'https://drive.google.com/uc?export=download&id=1MDibtvfL4ncRaT-K3_XCjBhhmrMeT3E1'
        elif Name_Dataset == 'Dev':
            file_name = 'subtaskB_dev.jsonl'
            url = 'https://drive.google.com/uc?export=download&id=19gWmqRh6v4ZqfdGmOZSHn6oQl21MPhWf'
        else:
            print('Invalid Dataset name')

        self.path = os.path.join(get_path('Dataset'), file_name)

        if Download:
            # Download file
            gdown.download(url, self.path, quiet=True)

    def read_jsonl_file(self, file_name, Dataset_Name, Percentage, Seed):
        """
        Read a JSON Lines file and load it into a dataset.

        Parameters:
        - file_name (str): The name of the JSON file to be read.

        Returns:
        - dataset (list): A list of dictionaries representing the dataset loaded from the JSON Lines file.
          Each dictionary contains 'text' and 'label' fields.
        """
        
 
        if Dataset_Name=='Train':
            labels_list = np.random.RandomState(Seed).permutation(int(71027))[:int(71027*(Percentage/100))]
        else:
            labels_list = np.arange(3000)

        file_path = os.path.join(get_path('Dataset'), file_name)

        dataset = []

        with open(file_path, "r") as file:
            for idx, line in enumerate(file):
                data = json.loads(line)

                label = data['label']
                text = data['text']
                model = data['model']
                labeled = (idx in labels_list)

                dataset.append({'text': text, 'label': label, 'model': model, 'labeled': labeled})

        return dataset

def get_path(folder_name):
    """
    Get the absolute path of a specified folder within the parent directory.

    Parameters:
    - folder_name (str): Name of the folder.

    Returns:
    - str: Absolute path of the specified folder within the parent directory.
    """
    # Get the path of the parent directory
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    folder_path = os.path.join(project_folder, folder_name)

    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def distribution_plot(train_dataset, val_dataset, output_filename='distribution_plot.png'):
    """
    Create distribution plots for the samples in each label category for both training and validation datasets.
    Save the plots to the plot folder.

    Parameters:
    - train_dataset: Training Datase object.
    - val_dataset: Validation Datase object.
    - output_filename: Filename for the saved plot (default is 'distribution_plot.png').
    """
    id2label = {0 :'human', 1 :'chatGPT', 2 :'cohere', 3 :'davinci', 4 :'bloomz', 5 :'dolly'}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    # Plot for Training Dataset
    train_labels = ['unlabeled' if data['labeled'] == False else id2label[data['label']] for data in train_dataset]
   
    train_label_counts = Counter(train_labels)

    train_color_palette = sns.color_palette("husl", n_colors=len(train_label_counts))

    sns.countplot(x=train_labels, palette=train_color_palette, hue=train_labels, ax=axes[0], legend=False)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title(f'Distribution of Samples for Each Label in Training Dataset')

    # Plot for Validation Dataset
    val_labels = [data['label'] for data in val_dataset]
    val_labels = [id2label[label] for label in val_labels]
    val_label_counts = Counter(val_labels)

    val_color_palette = sns.color_palette("husl", n_colors=len(val_label_counts))

    sns.countplot(x=val_labels, palette=val_color_palette, hue=val_labels, ax=axes[1], legend=False)
    axes[1].set_xlabel('Labels')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title(f'Distribution of Samples for Each Label in Validation Dataset')

    # Save the plot
    output_file_path = os.path.join(get_path('Plots'), output_filename)
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

def initialize_model_Bert(model_name, device, requires_grad=True):
    """
    Initialize a BERT model for sequence classification.

    Args:
        model_name (str): The name of the BERT model to be used.
        device (torch.device): The device 
        requires_grad (bool): Whether to set requires_grad for model parameters. Default is True.

    Returns:
        BertForSequenceClassification: The initialized BERT model.
    """
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    model = BertForSequenceClassification.from_pretrained(model_name).to(device)
    
    for param in model.parameters():
        param.requires_grad = requires_grad
    
    return model


class model_train:
    def __init__(self,model_Bert,Discriminator,Generator,kind_G,BERT_Block,optimizer_D,optimizer_G,
                 criterion_sup,lambda_sup,lambda_feat,criterion_G,tokenizer,train_loader,val_loader,num_epochs,device,d_in,Name, Use_save_model=False ,save_model= True):
        self.model_Bert = model_Bert
        self.Discriminator = Discriminator
        self.Generator = Generator
        self.kind_G = kind_G
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.criterion_sup = criterion_sup
        self.criterion_G = criterion_G
        self.lambda_sup = lambda_sup
        self.lambda_feat = lambda_feat
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.Name = Name
        self.BERT_Block = BERT_Block
        self.d_in = d_in
        self.save_model = save_model
        
        if self.save_model or Use_save_model:
            self.path_save_model = '/content/drive/MyDrive/saved_models'
            os.makedirs(self.path_save_model, exist_ok=True)

        if Use_save_model:
            self.model_Bert.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_model_Bert.pth')))
            self.Discriminator.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_Discriminator.pth')))
            if self.kind_G == 'G2':
                self.Generator.model_bert.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_Generator.pth')))
            else:
                self.Generator.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_Generator.pth')))
            print('saved Model loaded')

    
    def train(self):
         
        real_label = 0
        fake_label = 1

        G_losses = []
        D_losses = []
        val_acces = []
        train_acces = []

        for epoch in range(self.num_epochs):
            # Training phase
            self.model_Bert.train()
            self.Discriminator.train()
            #self.Generator.train()

            G_loss = 0
            D_loss = 0

            for data in tqdm.tqdm(self.train_loader):
                texts = data['text']
                labels = data['label'].to(self.device)
                prompt = data['model']
                labeled = data['labeled']

                real_batch_size = labels.shape[0]

                do_we_have_unlabeleds = sum(labeled)

                # Train Discriminator
                
                CLS = self.BERT_Block(text= texts, tokenizer=self.tokenizer, model=self.model_Bert, device =self.device)

                ## real data
                label_GAN_real = torch.full((real_batch_size,), real_label, dtype=torch.float, device=self.device)
                logits_real, features_real = self.Discriminator(CLS)

                loss_D_unsup_real = self.criterion_G(logits_real[:,0], label_GAN_real)

                if do_we_have_unlabeleds != 0:
                    loss_D_sup = self.criterion_sup(logits_real[:, 1:][labeled], labels[labeled])
                else:
                    loss_D_sup = 0

                
                ## fake data
                if self.kind_G == 'G1':
                        noise = torch.randn(real_batch_size, self.d_in, device=self.device)
                        v_G = self.Generator(noise)
                elif self.kind_G == 'G2':
                        v_G = self.Generator.generate(prompt)
                else:
                    print('Invalid Generator')

                label_GAN_fake = torch.full((real_batch_size,), fake_label, dtype=torch.float, device=self.device)
                logits_fake,features_fake= self.Discriminator(v_G)
                loss_D_unsup_fake = self.criterion_G(logits_fake[:,0], label_GAN_fake)

                # Total loss Discriminator
                loss_D = loss_D_unsup_real + loss_D_unsup_fake + loss_D_sup * self.lambda_sup 
            

                ## Train Generator
                
                loss_G_unsup = self.criterion_G(logits_fake[:,0], label_GAN_real)
                loss_G_feat = torch.mean(torch.square(torch.mean(features_real, dim=0) - torch.mean(features_fake, dim=0)))
                # Total Generator
                loss_G = loss_G_unsup + loss_G_feat * self.lambda_feat

                # Backward pass and optimization for Discriminator and Generator

                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()
                loss_G.backward(retain_graph=True)
                loss_D.backward()
                self.optimizer_D.step() 
                self.optimizer_G.step()

                G_loss += loss_G.item()
                D_loss += loss_D.item()
            
            G_losses.append(G_loss/len(self.train_loader))
            D_losses.append(D_loss/len(self.train_loader))

            # Calculate accuracy 
            val_acces.append(self.Accuracy(self.val_loader))
            #train_acces.append(self.Accuracy(self.train_loader))

            print(f'Epoch {epoch+1}/{self.num_epochs} | G_loss: {G_losses[-1]:.4f} | D_loss: {D_losses[-1]:.4f}  | Val_Accuracy: {100*val_acces[-1]:.2f}')

            # Save the best model
            if val_acces[-1] == max(val_acces) and self.save_model:
                torch.save(self.model_Bert.state_dict(), os.path.join(self.path_save_model, f'{self.Name}_model_Bert.pth'))
                torch.save(self.Discriminator.state_dict(), os.path.join(self.path_save_model, f'{self.Name}_Discriminator.pth'))
                if self.kind_G == 'G2':
                    torch.save(self.Generator.model_bert.state_dict(), os.path.join(self.path_save_model, f'{self.Name}_Generator.pth'))
                else:
                    torch.save(self.Generator.state_dict(), os.path.join(self.path_save_model, f'{self.Name}_Generator.pth'))
                print('Best Model saved to Google Drive')  
          
        # Plotting losses
        # Generator
        plt.figure(figsize=(12, 4))
        plt.suptitle(f'{self.Name} Model Summary', fontsize=16)

        plt.subplot(1, 3, 1)
        plt.plot(G_losses,label='Generator')
        plt.title('Generator Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Discriminator
        plt.subplot(1, 3, 2)
        plt.plot(D_losses,label='Discriminator')
        plt.title('Discriminator Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracies
        plt.subplot(1, 3, 3)
        plt.plot(val_acces, label = 'Validation')
        #plt.plot(train_acces, label = 'Train')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        output_file_path = os.path.join(get_path('Plots'), f'{self.Name} Model Summary')
        plt.savefig(output_file_path)
        plt.show()
    
    def Accuracy(self, loader, Use_best_model=False):
        if Use_best_model:
            self.model_Bert.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_model_Bert.pth')))
            self.Discriminator.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_Discriminator.pth')))
            if self.kind_G == 'G2':
                self.Generator.model_bert.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_Generator.pth')))
            else:
                self.Generator.load_state_dict(torch.load(os.path.join(self.path_save_model, f'{self.Name}_Generator.pth')))
            print('saved Best_Model loaded')

        self.model_Bert.eval()
        self.Discriminator.eval()
        correct = 0
        with torch.no_grad():
            for data in loader:
                texts = data['text']
                labels = data['label'].to(self.device)

                CLS = self.BERT_Block(text= texts, tokenizer=self.tokenizer, model=self.model_Bert, device=self.device)
                logits,_ = self.Discriminator(CLS)
                
                
                pred=logits[:,1:].argmax(dim=1,keepdim=True)
                correct +=pred.eq(labels.view_as(pred)).sum().item()

        return correct / (loader.dataset.__len__())


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
    plt.savefig(f'../Plots/{label}.png')
    plt.show()