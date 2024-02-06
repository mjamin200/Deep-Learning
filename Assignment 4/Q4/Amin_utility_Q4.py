import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os






class model_train:
    def __init__(self, model, Name, train_loader, test_loader, val_loader, optimizer, criterion, num_epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device
        self.val_loader = val_loader
        self.Name = Name


    def train(self):

        folder_path = 'saved_models'
        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Training and validation loop
        train_losses=[]
        val_losses=[]

        self.all_codebook_vectors=[]
        


        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            val_loss=0.0
            
            for inputs, _ in tqdm.tqdm(self.train_loader):

                inputs = inputs.to(self.device).to(torch.float32)

                self.optimizer.zero_grad()
                outputs,vq_loss = self.model(inputs)

                reconstruction_loss=self.criterion(outputs.to(torch.float32),inputs)

                loss = reconstruction_loss + vq_loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
            train_losses.append(train_loss/len(self.train_loader))

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                 for inputs, _ in self.val_loader:
                    inputs = inputs.to(self.device).to(torch.float32)
                    outputs,vq_loss = self.model(inputs)

                    reconstruction_loss=self.criterion(outputs.to(torch.float32),inputs)
                    loss = reconstruction_loss + vq_loss
                    val_loss += loss.item()
                    

            val_losses.append(val_loss / len(self.val_loader))

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
            self.all_codebook_vectors.append(self.model.vq_layer.embedding.weight.data.cpu().numpy())
            # save best model
            if val_losses[-1]==min(val_losses):
                self.model_path = os.path.join(folder_path, f'best_model_{self.Name}.pth')
                torch.save(self.model.state_dict(), self.model_path)
                print("Saved best model")
        
        # Plotting losses
        plt.figure(figsize=(12, 4))
        plt.suptitle(f'{self.Name} Model', fontsize=16)

        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()


        
    def Show_test_Image(self,Num_sample=8):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        with torch.no_grad():
            batch = next(iter(self.test_loader))
            data, _ = batch
            data = data.to(self.device)

            z = self.model.encoder(data)
            z,_=self.model.vq_layer(z)
            out = self.model.decoder(z)

            z = z.mean(dim=1)  # for showing z make num_channel = 1 
            

            if Num_sample> out.shape[0]:
              print('Num samples must be lower then test batch size')
              Num_sample = out.shape[0]

            
            fig, axes = plt.subplots(3, Num_sample, figsize=(16, 5))
            plt.suptitle("Images", fontsize=16)

            for i in range(Num_sample) :
                #print(type(z[i]))
                img = out[i].cpu().numpy()
                original = data[i].cpu().numpy()
                codebook = z[i].unsqueeze(0).cpu().numpy()
                
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img, 0, 1)

                original = np.transpose(original, (1, 2, 0))
                original = np.clip(original, 0, 1)

                codebook = np.transpose(codebook, (1, 2, 0))
                codebook = np.clip(codebook, 0, 1)

                axes[0, i].imshow(original)
                axes[0, i].set_title("Original")
                axes[0, i].axis("off")

                axes[1, i].imshow(img)
                axes[1, i].set_title("Reconstruction")
                axes[1, i].axis("off")

                axes[2, i].imshow(codebook)
                axes[2, i].set_title("Codebook")
                axes[2, i].axis("off")

            plt.tight_layout()
            plt.show()  
            print('\n To display the codeBook output, set num_channel to 1, along with the meanings for all 256 channels \n')


            vectors=self.model.vq_layer.embedding.weight.data.cpu().numpy()
            # plot codebook vectors with dim = 2
            if vectors.shape[1] == 2:
                fig, ax = plt.subplots()
                colors = ['r', 'g', 'b']

                for v, c in zip(vectors, colors):
                    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c)

                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-.5, .5])

                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("Codebook Vectors")
                plt.show()

    def make_gif(self):

        if self.all_codebook_vectors[0].shape[1] != 2:
            print ('invalid dimension making gif')
            return  

        colors = ['r', 'g', 'b']
        images = []

        #print(self.all_codebook_vectors)

        for i, vectors in enumerate(self.all_codebook_vectors):
            fig, ax = plt.subplots()
            ax.set_xlim([-0.6, 0.6])
            ax.set_ylim([-0.6, 0.6])
            
            # Plot vectors in each frame
            for v, c in zip(vectors, colors):
                ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c)
            
            plt.grid()
            filename = f'frame_{i}.png'
            plt.savefig(filename)
            plt.close()
            
            images.append(imageio.imread(filename))

        # Create a GIF from the images
        imageio.mimsave('Codebook Vectors.gif', images, fps=2)
        for i in range(len(self.all_codebook_vectors)):
            os.remove(f'frame_{i}.png')
        

      