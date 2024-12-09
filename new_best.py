import random
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import zipfile
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CFG = {
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(1024 + 512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    '''
    def conv_block(self, in_channels, out_channels):  # BatchNorm 대신 GroupNorm
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm 추가
            nn.ReLU(inplace=True),
        )
    '''
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        e5 = self.enc5(nn.MaxPool2d(2)(e4))

        d1 = self.dec1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.sigmoid(self.final(d4))

# BatchNorm 대신 GroupNorm
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),  # GroupNorm 추가
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),  # GroupNorm 추가
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),  # GroupNorm 추가
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

'''
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    '''

class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None): #, limit=5000
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        
        '''
        assert len(self.input_images) == len(self.gt_images), "The number of images in gray and color folders must match"
    
        if limit:
            self.input_images = self.input_images[:limit]
            self.gt_images = self.gt_images[:limit]
        '''

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        )

generator = UNet().to(device)
discriminator = PatchGANDiscriminator().to(device)
# generator = nn.DataParallel(generator, device_ids=[0, 1, 2, 3])
# discriminator = nn.DataParallel(discriminator, device_ids=[0, 1, 2, 3])

# 데이터셋 및 DataLoader 생성
dataset = ImageDataset("/home/work/.dacon/opendata/train_input", "/home/work/.dacon/opendata/train_gt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

# 데이터셋 분리 #이거 추가함
train_input, val_input = train_test_split(sorted(os.listdir("/home/work/.dacon/opendata/train_input")), test_size=0.2, random_state=CFG['SEED'])
train_gt, val_gt = train_test_split(sorted(os.listdir("/home/work/.dacon/opendata/train_gt")), test_size=0.2, random_state=CFG['SEED'])

train_dataset = ImageDataset("/home/work/.dacon/opendata/train_input", "/home/work/.dacon/opendata/train_gt")
val_dataset = ImageDataset("/home/work/.dacon/opendata/train_input", "/home/work/.dacon/opendata/train_gt") #limit 추가 (, limit=1000)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

# 손실 함수 및 옵티마이저 설정
adversarial_loss = nn.BCEWithLogitsLoss()  #원래 nn.BCELoss
pixel_loss = nn.L1Loss()  #원래 MSELoss

optimizer_G = optim.RMSprop(generator.parameters(), lr=1e-4) #원래 Adam
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=1e-4) #원래 Adam

'''
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999)) #크게
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)) #너무 작으면 학습 느려짐
'''

# 학습률 스케줄러 추가
scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=3, verbose=True)
scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=3, verbose=True)

epochs = 100
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = "checkpoint.pth"

for epoch in range(epochs):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for input_images, gt_images in train_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            real_labels = torch.ones_like(discriminator(gt_images)).to(device)
            fake_labels = torch.zeros_like(discriminator(input_images)).to(device)

            optimizer_G.zero_grad()
            fake_images = generator(input_images)
            pred_fake = discriminator(fake_images)

            g_loss_adv = adversarial_loss(pred_fake, real_labels)
            g_loss_pixel = pixel_loss(fake_images, gt_images)
            g_loss = g_loss_adv + 100 * g_loss_pixel
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            pred_real = discriminator(gt_images)
            loss_real = adversarial_loss(pred_real, real_labels)

            pred_fake = discriminator(fake_images.detach())
            loss_fake = adversarial_loss(pred_fake, fake_labels)

            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            pbar.set_postfix(generator_loss=g_loss.item(), discriminator_loss=d_loss.item())
            pbar.update(1)

    print(f"Epoch [{epoch+1}/{epochs}] - Generator Loss: {running_loss_G / len(train_loader):.4f}, Discriminator Loss: {running_loss_D / len(train_loader):.4f}")

    test_input_dir = "/home/work/.dacon/opendata/test_input"
    output_dir = f"output_images_epoch_{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for img_name in sorted(os.listdir(test_input_dir)):
            img_path = os.path.join(test_input_dir, img_name)
            img = cv2.imread(img_path)
            input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            output = generator(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
            output = output.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, img_name), output)

    zip_filename = "/home/work/.dacon/result/13_final.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_name in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, img_name), arcname=img_name)
    print(f"All images saved in {zip_filename}")

    # 체크포인트 저장
    #checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)
print(f"Checkpoint saved at {checkpoint_path}")

generator.train()  
discriminator.train()  
