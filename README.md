# dacon

#  AI프로젝트II - 이미지 색상화 및 손실 부분 복원 AI 경진대회

### 목차

+ [I. 데이터 처리](https://github.com/jijeongwon/dacon/blob/main/README.md#i-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%B2%98%EB%A6%AC)
+ [II. 사용 모델](https://github.com/jijeongwon/dacon/blob/main/README.md#ii-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85)
+ [III. 그 외 코드](https://github.com/jijeongwon/dacon/blob/main/README.md#iii-%EC%BD%94%EB%93%9C)
+ [IV. 실험 결과](https://github.com/jijeongwon/dacon/blob/main/README.md#iv-%EC%8B%A4%ED%97%98-%EA%B2%B0%EA%B3%BC)
+ [V. 결론](https://github.com/jijeongwon/dacon/blob/main/README.md#v-%EA%B2%B0%EB%A1%A0)

***

## I. 데이터 처리

   #### 1. 데이터셋 소개

 데이터셋은 Dacon에서 제공된 경진대회 전용 데이터셋을 사용했다. 데이터셋은 두 부분으로 나뉘며, 하나는 train_input: 흑백, 일부 손상된 PNG 학습 이미지 (input, 29603장)이고, 다른 하나는 train_gt: 원본 PNG 이미지 (target, 29603장)로 구성되어 있다.

   #### 2. 데이터 전처리

 기존의 resize, totensor, normalize 방법 외에도 아래의 여러 가지 전처리 방법을 적용시켜보았다.
 
 (전처리 하였을 때 성능 훨씬 안 좋고 눈으로 봐도 이상해서 결국 n번째 실험 부터는 transform 모두 제거........)

 ```
    # 데이터 전처리 설정
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    ])
```

<img width="30%" src="https://github.com/user-attachments/assets/35d2245d-aa04-401b-b236-ae2be9ac977f"/>
<img width="30%" src="https://github.com/user-attachments/assets/d8be503a-aac9-4808-8f8a-eca313c957bb"/>
<img width="30%" src="https://github.com/user-attachments/assets/b6912b7f-66a9-46bc-bd70-afe9fe2eb3ac"/>


***
 
   #### 3. 데이터 분할

데이터셋을 train과 validation으로 분할하였다. 전체 데이터의 80%는 train에 사용하였고, 나머지 20%는 validation에 사용하였다.

```
    # 데이터셋 분리
    train_input, val_input = train_test_split(sorted(os.listdir("/home/work/.dacon/opendata/train_input")), test_size=0.2, random_state=CFG['SEED'])
    train_gt, val_gt = train_test_split(sorted(os.listdir("/home/work/.dacon/opendata/train_gt")), test_size=0.2, random_state=CFG['SEED'])
```

***

## II. 사용 모델

우선 Generator 모델이다. 여기서는 "Spectral Normalizatoin", "Group Normalization", "각 블록마다 Batch Normalization", "다운샘플링/업샘플링 블록 분리" 등의 기능을 추가해보았다. 

그 중 가장 성능이 좋았던 모델구조는 이렇다.

+ **U-Net 기반 Generator**

```
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
```

기존 baseline 코드에서 'batch_norm'을 'group_norm'으로 변경하였고, 'down_sampling블록'과 'up_sampling블록'을 분리하였다.


다음은 Discriminator 모델이다. 여기서는 "Spectral Normalizatoin", "Group Normalization" 등의 기능을 추가해보았다.

Generator 모델에서와 마찬가지로, 기존 baseline 코드의 Discriminator 모델에서 "Group Normalization"만 추가했을 때 가장 성능이 좋았다.

+ **PatchGAN Discriminator**
```
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
```

## III. 그 외 시도해본 것들 (limit, scheduler, loss, Optimizer, ...)

```
    #scheduler 추가
    scheduler = StepLR(step_size=10, gamma=0.5)
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=3, verbose=True)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=3, verbose=True)
    
    #loss 변경
    adversarial_loss = nn.BCEWithLogitsLoss()  #원래 nn.BCELoss
    pixel_loss = nn.L1Loss()  #원래 MSELoss

    #optimizer 변경
    optimizer_G = optim.RMSprop(generator.parameters(), lr=1e-4) #원래 Adam
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=1e-4) #원래 Adam
```



***


## IV. 실험 결과

    -----------------------------------------------
    baseline - submission_1
    public score: 0.3396743653
    -----------------------------------------------
    baseline + transform/scheduler - submission_4
    public score: 0.1303616413
    -----------------------------------------------
    UNet+PatchGAN - submission_8
    public score: 0.3396743653
    -----------------------------------------------
    UNet+PatchGAN + BCEWithLogitsLoss/L1Loss
    + RMSProp, ReduceLROnPlateau - submission_best
    Training R-squared: 0.7081650991544655
    Testing R-squared: 0.7983571856679336
    Mean Absolute Error: 2082.1281143876554
    -----------------------------------------------

    
    -----------------------------------------------



## V. 결론

+ **전체적인 submission을 순서대로 보여주는 표이다.** 

<img width="90%" src="https://github.com/user-attachments/assets/05122db1-61f3-4411-9a6d-ce6b189f7276"/>

limit을 걸고 학습했을 때와 걸지 않고 학습했을 때 Public Score 차이가 많이 나는 것을 알 수 있다.

또한, transform을 진행하지 않은 데이터셋을 활용하여 GroupNorm을 추가함으로써 모델의 구조를 변형하고 GAN 학습에 적절한 loss를 조합했을 때 가장 이미지 복원 성능이 좋은 것을 확인할 수 있다.




