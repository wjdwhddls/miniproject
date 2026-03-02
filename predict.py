import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. 모델 뼈대(설계도) 가져오기 (train.py에 있던 것과 똑같이 생겨야 합니다)
class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False) # 이미 학습된 걸 덮어씌울 거라 False!
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.class_head = nn.Linear(num_features, num_classes)
        self.protein_head = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        out_class = self.class_head(features)
        out_protein = self.protein_head(features)
        return out_class, out_protein

# 2. 아무것도 모르는 빈 깡통 모델 객체 생성
model = MultiHeadResNet(num_classes=3)

# 3. 아까 저장해둔 '똑똑해진 뇌(가중치)' 파일 불러와서 깡통 모델에 덮어씌우기!
model.load_state_dict(torch.load('beef_ai_brain.pth'))
model.eval() # 뇌세포 고정 (시험 모드 켜기)

print("✅ 똑똑한 AI 뇌 이식 완료! 실전 테스트를 준비합니다.\n")

# 4. 이미지 변환 규칙 (학습 때와 완벽히 동일해야 함)
my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. 테스트할 고기 사진 예측하기
class_names = {0: "안심", 1: "우둔살", 2: "부채살"}
test_image_path = 'images/28.jpeg' # 여기서 2.jpg, 3.jpg로 바꿔가며 테스트해보세요!

image = Image.open(test_image_path).convert("RGB")
input_batch = my_transform(image).unsqueeze(0)

with torch.no_grad(): # 역전파(학습) 끄기
    pred_class, pred_protein = model(input_batch)
    
    _, predicted_idx = torch.max(pred_class, 1)
    predicted_name = class_names[predicted_idx.item()]
    predicted_protein_value = pred_protein.item()

print(f"📸 입력 사진: {test_image_path}")
print(f"🥩 AI의 예측 부위: {predicted_name}")
print(f"💪 AI의 예측 단백질(100g당): {predicted_protein_value:.1f}g")