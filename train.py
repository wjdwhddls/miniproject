# import os
# import pandas as pd
# from PIL import Image
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.models as models
# import torchvision.transforms as transforms
# import torch.optim as optim

# # =====================================================================
# # [1] 컨베이어 벨트 설계 (Dataset 상속)
# # 파이토치의 기본 Dataset 클래스를 상속받아 나만의 데이터 규칙을 만듭니다.
# # =====================================================================
# class BeefProteinDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         # 1. 아까 만든 CSV 파일(정답지)을 팬더스(pandas)로 읽어옵니다.
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         # 전체 데이터가 몇 장인지 반환합니다.
#         return len(self.annotations)

#     def __getitem__(self, index):
#         # 특정 인덱스의 사진과 정답을 하나로 묶어서 반환합니다.
#         img_name = self.annotations.iloc[index, 0] # 파일 이름
#         img_path = os.path.join(self.img_dir, img_name)
        
#         # 사진을 열고 RGB 형태로 강제 통일시킵니다 (확장자 문제 완벽 해결!)
#         image = Image.open(img_path).convert("RGB")
        
#         # 정답(부위, 단백질량) 가져오기
#         y_class = self.annotations.iloc[index, 1]  
#         y_protein = self.annotations.iloc[index, 2] 
        
#         # 이미지를 파이토치 텐서(숫자 배열)로 변환
#         if self.transform:
#             image = self.transform(image)
            
#         # 정답들도 파이토치 텐서로 변환
#         y_class = torch.tensor(y_class, dtype=torch.long)
#         y_protein = torch.tensor(y_protein, dtype=torch.float32)
        
#         return image, y_class, y_protein


# # =====================================================================
# # [2] 멀티 헤드 모델 설계 (nn.Module 상속)
# # 파이토치의 신경망 기본 클래스인 nn.Module을 상속받아 구조를 튜닝합니다.
# # =====================================================================
# class MultiHeadResNet(nn.Module):
#     def __init__(self, num_classes):
#         super(MultiHeadResNet, self).__init__()
#         # 1. 똘똘한 ResNet18 몸통을 가져옵니다. (미리 학습된 가중치 사용)
#         self.backbone = models.resnet18(pretrained=True)
        
#         # 2. 기존의 마지막 출구 크기를 확인하고, 원래 있던 출구를 부숴버립니다.
#         num_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Identity()
        
#         # 3. 새로운 두 개의 출구(Head)를 달아줍니다.
#         self.class_head = nn.Linear(num_features, num_classes) # 부위 분류용
#         self.protein_head = nn.Linear(num_features, 1)         # 단백질 예측용

#     def forward(self, x):
#         # 데이터가 모델을 통과하는 길(순전파)을 정의합니다.
#         features = self.backbone(x)
        
#         out_class = self.class_head(features)
#         out_protein = self.protein_head(features)
        
#         return out_class, out_protein


# # =====================================================================
# # [3] 잘 작동하는지 테스트해 보기! (메인 실행부)
# # =====================================================================
# if __name__ == "__main__":
#     # ResNet이 좋아하는 이미지 변환 규칙 (224x224 사이즈로 맞추기)
#     my_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 1. 데이터셋 객체 생성 (부위가 3개라고 가정)
#     dataset = BeefProteinDataset(csv_file='beef_dataset.csv', 
#                                  img_dir='images', 
#                                  transform=my_transform)
    
#     # 2. 컨베이어 벨트(DataLoader) 생성 (한 번에 4장씩 묶어서 처리)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     # 3. 내 커스텀 모델 객체 생성 (분류할 부위 개수 = 3)
#     model = MultiHeadResNet(num_classes=3)

#     print("\n✅ 컨베이어 벨트와 모델 세팅이 완료되었습니다!")
#     print("데이터를 모델에 한 번 넣어보겠습니다...\n")

#     # 4. 데이터로더에서 데이터 한 묶음(4장)만 뽑아서 모델에 넣어보기
#     for images, labels, proteins in dataloader:
#         pred_classes, pred_proteins = model(images)
        
#         print(f"👉 입력 이미지 형태: {images.shape} (배치크기, 채널, 가로, 세로)")
#         print(f"👉 부위 예측 결과 형태: {pred_classes.shape} (배치크기, 클래스개수)")
#         print(f"👉 단백질 예측 결과 형태: {pred_proteins.shape} (배치크기, 1)")
#         break # 한 번만 테스트하고 종료
    
# # =====================================================================
# # [4] 본격적인 학습(Training) 루프 만들기
# # =====================================================================
# print("\n🚀 본격적인 학습을 시작합니다!")

# # 1. 채점 기준(Loss Function) 준비
# criterion_class = nn.CrossEntropyLoss() # 객관식 채점용 (부위 분류)
# criterion_protein = nn.MSELoss()        # 주관식 채점용 (단백질 예측)

# # 2. 학습 도구(Optimizer) 준비 (보통 Adam을 가장 많이 씁니다)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 3. 전체 데이터를 몇 번 반복해서 공부할 것인가? (Epoch)
# num_epochs = 5 

# for epoch in range(num_epochs):
#     epoch_loss = 0.0 # 이번 회차의 전체 오차
    
#     # 컨베이어 벨트(dataloader)에서 4장씩 꺼내오며 반복
#     for i, (images, true_classes, true_proteins) in enumerate(dataloader):
        
#         # [STEP 1] 이전 문제의 오답 노트 기록을 지워줍니다.
#         optimizer.zero_grad()
        
#         # [STEP 2] 순전파: 모델에게 사진을 주고 예측해 보라고 합니다.
#         pred_classes, pred_proteins = model(images)
        
#         # [STEP 3] 오차 계산: 정답과 얼마나 차이나는지 채점합니다.
#         # 단백질 예측값의 형태를 [4, 1]에서 [4]로 맞춰주기 위해 squeeze()를 씁니다.
#         loss_class = criterion_class(pred_classes, true_classes)
#         loss_protein = criterion_protein(pred_proteins.squeeze(), true_proteins)
        
#         # 두 오차를 합쳐서 최종 오차를 구합니다! (멀티 헤드의 핵심)
#         total_loss = loss_class + loss_protein
        
#         # [STEP 4] 역전파: 오차를 줄이기 위해 어느 방향으로 수정해야 할지 계산합니다.
#         total_loss.backward()
        
#         # [STEP 5] 가중치 업데이트: 실제로 모델의 뇌 구조를 살짝 수정합니다.
#         optimizer.step()
        
#         epoch_loss += total_loss.item()
        
#     # 한 번의 Epoch(전체 데이터 1회독)가 끝날 때마다 평균 오차를 출력합니다.
#     avg_loss = epoch_loss / len(dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] 완료! -> 현재 평균 오차(Loss): {avg_loss:.4f}")

# print("\n🎉 모든 학습이 완료되었습니다! AI가 똑똑해졌습니다!")

# # =====================================================================
# # [추가] 학습된 뇌(가중치)를 파일로 영구 저장하기
# # =====================================================================
# torch.save(model.state_dict(), 'beef_ai_brain.pth')
# print("💾 AI의 뇌 상태가 'beef_ai_brain.pth' 파일로 안전하게 저장되었습니다!")

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# =====================================================================
# [1] 컨베이어 벨트 설계 (Dataset) - 이전과 동일
# =====================================================================
class BeefProteinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        y_class = self.annotations.iloc[index, 1]  
        y_protein = self.annotations.iloc[index, 2] 
        
        if self.transform:
            image = self.transform(image)
            
        y_class = torch.tensor(y_class, dtype=torch.long)
        y_protein = torch.tensor(y_protein, dtype=torch.float32)
        
        return image, y_class, y_protein

# =====================================================================
# [2] 튜닝된 멀티 헤드 모델 설계 (Dropout, ReLU 적용!)
# =====================================================================
class TunedMultiHeadResNet(nn.Module):
    def __init__(self, num_classes):
        super(TunedMultiHeadResNet, self).__init__()
        
        # 💡 경고 메시지(Warning) 해결: weights='DEFAULT' 사용
        # (ResNet50을 쓰고 싶다면 models.resnet50(weights='DEFAULT')로 바꾸기만 하면 됩니다!)
        self.backbone = models.resnet18(weights='DEFAULT')
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 💡 튜닝 1: 부위 분류 머리 (과적합 방지)
        self.class_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5), # 뇌세포 50% 기절!
            nn.Linear(256, num_classes)
        )
        
        # 💡 튜닝 2: 단백질 예측 머리 (과적합 방지)
        self.protein_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3), # 뇌세포 30% 기절!
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out_class = self.class_head(features)
        out_protein = self.protein_head(features)
        return out_class, out_protein

# =====================================================================
# [3] 메인 실행부 및 본격적인 학습 루프
# =====================================================================
if __name__ == "__main__":
    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BeefProteinDataset(csv_file='beef_dataset.csv', img_dir='images', transform=my_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 튜닝된 모델 객체 생성
    model = TunedMultiHeadResNet(num_classes=3)

    criterion_class = nn.CrossEntropyLoss() 
    criterion_protein = nn.MSELoss()        
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 💡 학습 횟수(Epoch)를 20번으로 넉넉하게 늘렸습니다! (Dropout이 지켜주니까 안심하세요)
    num_epochs = 20 

    print(f"\n🚀 총 {num_epochs}번의 튜닝된 모델 학습을 시작합니다!")

    for epoch in range(num_epochs):
        # 💡 매우 중요: 모델에게 "지금은 훈련 중이야!(Dropout 작동시켜!)"라고 알려주는 스위치
        model.train()
        epoch_loss = 0.0 
        
        for i, (images, true_classes, true_proteins) in enumerate(dataloader):
            optimizer.zero_grad()
            
            pred_classes, pred_proteins = model(images)
            
            loss_class = criterion_class(pred_classes, true_classes)
            loss_protein = criterion_protein(pred_proteins.squeeze(), true_proteins)
            
            total_loss = loss_class + loss_protein
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] 완료! -> 현재 평균 오차(Loss): {avg_loss:.4f}")

    print("\n🎉 모든 학습이 완료되었습니다! AI가 한층 더 똑똑해졌습니다!")

    # 똑똑해진 뇌 저장하기 (이름을 살짝 바꿨습니다)
    torch.save(model.state_dict(), 'beef_ai_brain_tuned.pth')
    print("💾 튜닝된 AI의 뇌 상태가 'beef_ai_brain_tuned.pth' 파일로 안전하게 저장되었습니다!")