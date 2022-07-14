import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import json, h5py, random

from pathlib import Path
from tqdm.notebook import tqdm

# =============================================================================================
# === COLLATE ANY FUNCTION ====================================================================
# =============================================================================================

def collate_any(batch):
   
    imgs  = torch.stack([r[0] for r in batch])
    texts = np.array([random.choice(r[1]) for r in batch])

    return imgs, texts

# =============================================================================================
# === CLASS FOR DATASET =======================================================================
# =============================================================================================

class CustomDataset(Dataset):
    def __init__(self, img_file, target_file, image_size=128):

        self.img_file = img_file
        self.captions = json.load(open(target_file, "r"))

        self.images   = None

        self.img_transform  = transforms.Compose([transforms.Resize(image_size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.CenterCrop(image_size)])

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
    
        if not self.images: self.images = h5py.File(self.img_file, 'r') 
            
        img = self.images["images"][idx].astype(float)
        img = torch.from_numpy((img - img.min()) / np.max([img.max() - img.min(), 1]))

        cap = self.captions[idx]       
                
        return self.img_transform(img), cap

    def __del__(self):
        if self.images:
            self.images.close()

# =============================================================================================
# === CLASS FOR TRAIN THE MODEL ===============================================================
# =============================================================================================

class ImagenTrainer(nn.Module):

    def __init__(self, imagen, epochs = 10, first_epoch=1, p=1, lr = 1e-4, eps = 1e-8,
                 beta1 = 0.9, beta2 = 0.99, device='cpu'):

        super(ImagenTrainer, self).__init__()

        self.p           = p
        self.imagen      = imagen
        self.unet        = imagen.unets[0]
        self.device      = device
        self.first_epoch = first_epoch
        self.optimizer   = Adam(self.unet.parameters(), lr=lr, eps=eps, betas=(beta1, beta2))
        self.epochs      = epochs

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.imagen.state_dict(), str(path))

    def load(self, path):

        path = Path(path)
        self.imagen.load_state_dict(torch.load(str(path)))

    @torch.no_grad()
    def sample(self, texts, cond_scale):
        
        output = self.imagen.sample(texts=texts, cond_scale=cond_scale)
        return output

    @torch.no_grad()
    def validation_loss(self, data):

        total_loss = 0.

        for images, texts in tqdm(data):

                images = images.to(self.device)

                loss = self.imagen(images, texts=texts, unet_number = 0, device=self.device)

                total_loss += loss.item()

        return total_loss/len(data)

    def forward(self, train_data, valid_data, path=None, inter_path=None, save_new_each=10):

        train_loss_per_epoch, valid_loss_per_epoch = [], []
        save_path = None

        for epoch in tqdm(range(self.first_epoch, self.epochs+1)):

            total_loss = 0.

            for images, texts in tqdm(train_data):

                images = images.to(self.device)

                self.optimizer.zero_grad()

                loss = self.imagen(images, texts=texts, unet_number = 0, device=self.device)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_train_loss = total_loss/len(train_data)
            total_valid_loss = self.validation_loss(valid_data)

            train_loss_per_epoch.append(total_train_loss)
            valid_loss_per_epoch.append(total_valid_loss)

            if path is not None:

                if (epoch - 1) % save_new_each == 0:
                    self.p += 1
                    save_path = path + str(self.p) + '.pth.tar'

                if save_path is not None:
                    self.save(save_path)
                else:
                    self.save(inter_path)

            print(f'Epoch: {epoch} | Train Loss: {total_train_loss} | Valid Loss: {total_valid_loss}')

        plt.plot(train_loss_per_epoch, label="Training")
        plt.plot(valid_loss_per_epoch, label="Validating")
        plt.title('Loss x Epoch')
        plt.show()


if __name__ == '__main__':

    pass