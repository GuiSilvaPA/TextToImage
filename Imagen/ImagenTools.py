import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import json, h5py, random

from pathlib import Path
from tqdm.notebook import tqdm

import numpy as np
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3

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
# === CLASS FOR DATASET OF HIGH RESOLUTION ====================================================
# =============================================================================================

class HighResolutionCustomDataset(Dataset):
    def __init__(self, img_file, target_file, image_size=64):

        self.captions = json.load(open(target_file, "r"))

        self.img_file      = img_file
        self.images        = None
        self.img_transform = transforms.Resize(image_size)

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
    
        if not self.images: self.images = h5py.File(self.img_file, 'r') 
            
        img = self.images["images"][idx].astype(float)
        img = torch.from_numpy((img - img.min()) / np.max([img.max() - img.min(), 1]))     
                
        return self.img_transform(img), img

    def __del__(self):
        if self.images:
            self.images.close()

# =============================================================================================
# === CLASS FOR TRAIN THE MODEL ===============================================================
# =============================================================================================

class ImagenTrainer(nn.Module):

    def __init__(self, imagen, high_model = None, epochs = 10, first_epoch=1, p=1, lr = 1e-4, eps = 1e-8,
                 beta1 = 0.9, beta2 = 0.99, device='cpu'):

        super(ImagenTrainer, self).__init__()

        self.p           = p
        self.imagen      = imagen
        self.unet        = imagen.unets[0]
        self.device      = device
        self.first_epoch = first_epoch
        self.optimizer   = Adam(self.unet.parameters(), lr=lr, eps=eps, betas=(beta1, beta2))
        self.epochs      = epochs
        self.high_model  = high_model

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.imagen.state_dict(), str(path))

    def load(self, path):

        path = Path(path)
        self.imagen.load_state_dict(torch.load(str(path)))

    def load_high(self, path):

        path = Path(path)
        self.imagen.load_state_dict(torch.load(str(path)))

    @torch.no_grad()
    def sample(self, texts, cond_scale, use_high_resolution = False):
        
        output = self.imagen.sample(texts=texts, cond_scale=cond_scale)

        if use_high_resolution:
            output = self.high_model(output)

        return output

    @torch.no_grad()
    def validation_loss(self, data):

        total_loss = 0.

        for images, texts in tqdm(data):

                images = images.float().to(self.device)

                loss = self.imagen(images, texts=texts, device=self.device)

                total_loss += loss.item()

        return total_loss/len(data)

    def forward(self, train_data, valid_data, path=None, inter_path=None, save_new_each=10):

        train_loss_per_epoch, valid_loss_per_epoch = [], []
        save_path = None

        for epoch in tqdm(range(self.first_epoch, self.epochs+1)):

            total_loss = 0.

            print(f'\n================================ EPOCH {epoch} ================================\n')

            for images, texts in tqdm(train_data):

                images = images.float().to(self.device)

                self.optimizer.zero_grad()

                loss = self.imagen(images, texts=texts, device=self.device)
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

            print(f'\nEpoch: {epoch} | Train Loss: {total_train_loss} | Valid Loss: {total_valid_loss}')

        plt.plot(train_loss_per_epoch, label="Training")
        plt.plot(valid_loss_per_epoch, label="Validating")
        plt.title('Loss x Epoch')
        plt.show()

# =============================================================================================
# === HIGH RESOLUTION TRAINER =================================================================
# =============================================================================================

class HighResolutionTrainer(nn.Module):

    def __init__(self, high_model, epochs = 5, lr = 1e-4, device='cpu'):
        super(HighResolutionTrainer, self).__init__()

        self.high_model = high_model.to(device)
        self.device     = device
        self.optimizer  = SGD(high_model.parameters(), lr=lr, weight_decay=0.0)
        self.criterion  = nn.L1Loss().to(device)
        self.epochs     = epochs        

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.high_model.state_dict(), str(path))

    @torch.no_grad()
    def validation_loss(self, data):

        total_loss = 0.

        for images, high_images in tqdm(data):

                images      = images.to(self.device).float()
                high_images = high_images.to(self.device).float()

                outputs = self.high_model(images)
                loss    = self.criterion(outputs.float(), high_images.float())

                total_loss += loss.item()*high_images.size(0)

        return total_loss/len(data)

    def forward(self, train_data, valid_data, path=None):

        train_loss_per_epoch, valid_loss_per_epoch = [], []

        for epoch in tqdm(range(1, self.epochs+1)):

            train_loss = 0.0
            self.high_model.train()

            print(f'\n================================ EPOCH {epoch} ================================\n')

            for images, high_images in tqdm(train_data):

                images      = images.to(self.device).float()
                high_images = high_images.to(self.device).float()

                self.optimizer.zero_grad()  
                
                # Forward
                outputs = self.high_model(images)
                loss    = self.criterion(outputs.float(), high_images.float())

                #Backward
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*high_images.size(0)

            total_train_loss = train_loss/len(train_data)
            total_valid_loss = self.validation_loss(valid_data)

            train_loss_per_epoch.append(total_train_loss)
            valid_loss_per_epoch.append(total_valid_loss)

            if path is not None:
                if total_valid_loss <= total_train_loss:
                    self.save(path)

            print(f'\nEpoch: {epoch} | Train Loss: {total_train_loss} | Valid Loss: {total_valid_loss}')

        plt.plot(train_loss_per_epoch, label="Training")
        plt.plot(valid_loss_per_epoch, label="Validating")
        plt.title('Loss x Epoch')
        plt.show()

# =============================================================================================
# === FID METRIC ==============================================================================
# =============================================================================================

def scale_images(images, new_shape):
    return np.asarray([resize(image, new_shape, 0) for image in images])
 
def FID(image, target_image, device='cpu'):
    
    # Based on: https://arxiv.org/pdf/1706.08500.pdf
    # FID = ||Mr - Mg||^2 + Trace(COVr + COVg + 2*(COVr*COVg)**(1/2))

    model        = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    image        = scale_images(image.cpu().numpy().transpose(0, 2, 3, 1), (299, 299))
    target_image = scale_images(target_image.cpu().numpy().transpose(0, 2, 3, 1), (299, 299))

    IMGg = model.predict(image)
    IMGr = model.predict(target_image)
        
    Mg, COVg = IMGg.mean(axis=0), np.cov(IMGg, rowvar=False)
    Mr, COVr = IMGr.mean(axis=0), np.cov(IMGr, rowvar=False)
    
    covmean = sqrtm(COVg.dot(COVr))    
    
    if np.iscomplexobj(covmean): covmean = covmean.real

    return np.sum((Mr - Mg)**2.0) + np.trace(COVg + COVr - 2.0 * covmean)

# =============================================================================================
# === GET SEQUENCE OF IMAGES ==================================================================
# =============================================================================================

def get_images(trainer, loader, num_imgs):

    imgs, texts = [], []

    for m, (i, t) in enumerate(loader):
        imgs.append(i[0])
        texts.append(t[0])
        if m+1 == num_imgs: break

    images_out = trainer.sample(texts=texts, cond_scale = 3.)

    for img, im, text in zip(images_out.cpu(), imgs, texts):

        img = img.cpu()
        fig = plt.figure(figsize=(10, 5))

        fig.add_subplot(1, 2, 1)
        plt.imshow((img.numpy().transpose(1, 2, 0)+1)/2)
        plt.title('Generated Image')
        plt.axis('off')

        fig.add_subplot(1, 2, 2)
        plt.imshow((im.numpy().transpose(1, 2, 0)+1)/2)
        plt.title('Reference Image')
        plt.axis('off')

        fig.suptitle(text)

        plt.show()

        print('\n=========================================================================== \n')

    return images_out, imgs, texts