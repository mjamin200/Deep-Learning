import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from torch.utils.data import Subset


class model_train_cross_validation:
    def __init__(self, model, Name, n_splits, train_dataset, test_dataset, batch_size, optimizer, criterion, num_epochs, device,Print_details=False,log_every = 5):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device
        self.Name = Name
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.Print_details = Print_details
        self.log_every = log_every
        self.train_dataset, self.test_dataset = train_dataset, test_dataset


    def train(self):

        kfold = KFold(n_splits=self.n_splits, shuffle=True)
        max_val_acc = 0.
        fold_train_losses=[]
        fold_train_acc=[]
        fold_val_losses=[]
        fold_val_acc=[]

        for fold, (train_indices, val_indices) in enumerate(kfold.split(self.train_dataset)):
            # Split data into train and test sets
            train_data = Subset(self.train_dataset, train_indices)
            val_data = Subset(self.train_dataset, val_indices)

            # Split data into train and test sets
            train_data = Subset(self.train_dataset, train_indices)
            val_data = Subset(self.train_dataset, val_indices)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

            if self.Print_details:
              print('\n','\n',f'Fold: {fold+1}',20*'######')

            # Training and validation loop
            train_losses=[]
            train_acc=[]
            val_losses=[]
            val_acc=[]


            for epoch in range(self.num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                val_loss=0.0
                train_correct = 0
                val_correct= 0

                for inputs, labels in (tqdm.tqdm(train_loader) if self.Print_details else train_loader):

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    outputs = outputs.squeeze() 
                    labels = labels.float()
                    loss = self.criterion(outputs , labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    
                    pred = outputs.round()
                    train_correct+=pred.eq(labels.view_as(pred)).sum().item()

                train_losses.append(train_loss/len(train_loader))
                train_acc.append(train_correct / len(train_loader.dataset))

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        outputs = outputs.squeeze() 
                        loss = self.criterion(outputs, labels.float())
                        val_loss += loss.item()
                        pred = outputs.round()
                        val_correct+=pred.eq(labels.view_as(pred)).sum().item()

                val_losses.append(val_loss / len(val_loader))
                val_acc.append(val_correct / len(val_loader.dataset))

                if self.Print_details and epoch % self.log_every == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train accuracy:{train_acc[-1]*100:.2f} Validation Loss: {val_losses[-1]:.4f}, Validation accuracy:{val_acc[-1]*100:.2f}")

                # save best model
                if max_val_acc < val_acc[-1]:
                    max_val_acc = val_acc[-1]
                    torch.save(self.model.state_dict(), f'best_model_{self.Name}.pth')
                    if self.Print_details:
                      print("Saved best model")

            if self.Print_details:
                # Plotting losses
                plt.figure(figsize=(12, 4))
                plt.suptitle(f'{self.Name} Model fold {fold+1}', fontsize=16)

                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Training Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.title('Training and Validation Losses')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # Plotting accuracies
                plt.subplot(1, 2, 2)
                plt.plot(train_acc, label='Training Accuracy')
                plt.plot(val_acc, label='Validation Accuracy')
                plt.title('Training and Validation Accuracies')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()

                plt.tight_layout()
                plt.show()

            fold_train_losses.append(train_losses)
            fold_train_acc.append(train_acc)
            fold_val_losses.append(val_losses)
            fold_val_acc.append(val_acc)




        # Plotting losses
        plt.figure(figsize=(16, 8))
        plt.suptitle(f'{self.Name} Model Summary', fontsize=16)

        plt.subplot(1, 2, 1)
        for i in range(self.n_splits):
            plt.plot(fold_val_losses[i], label=f'Validation Loss fold {i+1}')
        plt.title('Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting accuracies
        plt.subplot(1, 2, 2)

        for i in range(self.n_splits):
            plt.plot(fold_val_acc[i], label=f'Validation Accuracy fold {i+1}')
        plt.title('Validation Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def test(self):
        self.model.load_state_dict(torch.load(f'best_model_{self.Name}.pth'))
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        test_loss=0.0
        test_correct=0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels.float())
                test_loss += loss.item()
                pred = outputs.round()
                test_correct+=pred.eq(labels.view_as(pred)).sum().item()

        test_losses=test_loss / len(test_loader)
        test_acc=test_correct / len(test_loader.dataset)
        print(f"\n \n Test Loss: {test_losses:.4f}, Test accuracy:{test_acc*100:.2f}")         

class my_SmilesTokenizer:
    def __init__(self):
        self.vocabulary = set(['<PAD>'])
        self.encoder = LabelEncoder()

    def tokenize(self, smiles):
        i = 0
        tokens = []
        while i < len(smiles):
            if i < len(smiles) - 1 and smiles[i].isupper() and smiles[i+1].islower() and smiles[i+1] != 'c':
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
        self.vocabulary.update([token for token in tokens])
        return tokens

    def get_atoms(self, tokens):
      atoms = []
      for token in tokens:
          if len(token) == 1 and token.isalpha():
              atoms.append(token.upper())
          elif len(token) > 1 and token.isalpha():
              atoms.append(token)
      return atoms
    
    def one_hot_encode(self, token):
        self.encoder.fit(list(self.vocabulary))
        label = self.encoder.transform([token])
        one_hot_vector = torch.zeros(len(self.vocabulary))
        one_hot_vector[label] = 1
        return one_hot_vector

    def embedded_smiles(self, smiles, max_length):
        tokens = self.tokenize(smiles)
        length_tokens = len(tokens)

        if length_tokens < max_length:
            left_padding = (max_length - length_tokens) // 2
            right_padding = max_length - length_tokens - left_padding
            tokens = ['<PAD>'] * left_padding + tokens + ['<PAD>'] * right_padding
        else:
            tokens = tokens[:max_length]
        
        embedded_vectors = torch.stack([self.one_hot_encode(token) for token in tokens])
        
        return embedded_vectors
    
    def get_vocabulary(self):
        return sorted(list(self.vocabulary))
    



                    