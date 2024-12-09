# dacon

#  AI프로젝트II - 이미지 색상화 및 손실 부분 복원 AI 경진대회

### 목차

+ [I. 데이터 처리](https://github.com/jijeongwon/dacon/blob/main/README.md#i-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%B2%98%EB%A6%AC)
+ [II. 모델 설명](https://github.com/jijeongwon/dacon/blob/main/README.md#ii-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85)
+ [III. 코드](https://github.com/jijeongwon/dacon/blob/main/README.md#iii-%EC%BD%94%EB%93%9C)
+ [IV. 실험 결과](https://github.com/jijeongwon/dacon/blob/main/README.md#iv-%EC%8B%A4%ED%97%98-%EA%B2%B0%EA%B3%BC)
+ [V. 결론](https://github.com/jijeongwon/dacon/blob/main/README.md#v-%EA%B2%B0%EB%A1%A0)

***

## I. 데이터 처리

   #### 1. 데이터셋 소개

 데이터셋은 Dacon에서 제공된 경진대회 전용 데이터셋을 사용했다. 데이터셋은 두 부분으로 나뉘며, 하나는 train_input: 흑백, 일부 손상된 PNG 학습 이미지 (input, 29603장)이고, 다른 하나는 train_gt: 원본 PNG 이미지 (target, 29603장)로 구성되어 있다.

   #### 2. 데이터 전처리

 이미지 크기 조정, 정규화, 이미지 데이터 타입 변환, 데이터 증강 등의 여러 가지 전처리 방법을 적용시켜보았다.
 
    # 데이터 전처리 설정
    
    transform = transforms.Compose([
    transforms.Resize((256, 256)), # 기존
    transforms.ToTensor(), # 기존
    transforms.Normalize([0.5], [0.5], # 기존
    transforms.RandomHorizontalFlip(p=0.5), # 새로 추가
    transforms.RandomVerticalFlip(p=0.5), # 새로 추가
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
])

***
 
   #### 3. 데이터 분할

데이터셋을 train과 validation으로 분할하였다. 전체 데이터의 80%는 train에 사용하였고, 나머지 20%는 validation에 사용한다.

    # 데이터셋 분리
    
    train_input, val_input = train_test_split(sorted(os.listdir("/home/work/.dacon/opendata/train_input")), test_size=0.2, random_state=CFG['SEED'])
    train_gt, val_gt = train_test_split(sorted(os.listdir("/home/work/.dacon/opendata/train_gt")), test_size=0.2, random_state=CFG['SEED'])

***

## II. 모델 설명



## III. 코드



## IV. 실험 결과



## V. 결론



