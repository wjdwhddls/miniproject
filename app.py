# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image

# # =====================================================================
# # [1] 모델 뼈대 준비 (train.py, predict.py와 동일)
# # =====================================================================
# class MultiHeadResNet(nn.Module):
#     def __init__(self, num_classes):
#         super(MultiHeadResNet, self).__init__()
#         # 최신 버전에 맞춰 weights=None으로 경고 메시지 방지
#         self.backbone = models.resnet18(weights=None) 
#         num_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Identity()
#         self.class_head = nn.Linear(num_features, num_classes)
#         self.protein_head = nn.Linear(num_features, 1)

#     def forward(self, x):
#         features = self.backbone(x)
#         out_class = self.class_head(features)
#         out_protein = self.protein_head(features)
#         return out_class, out_protein

# # =====================================================================
# # [2] 뇌(가중치) 불러오기 함수 (최적화)
# # @st.cache_resource를 달아주면, 화면이 새로고침될 때마다 무거운 뇌를 
# # 다시 읽어오는 것을 방지하여 앱 속도가 엄청나게 빨라집니다.
# # =====================================================================
# @st.cache_resource
# def load_model():
#     model = MultiHeadResNet(num_classes=3)
#     # 맥북 환경 등 CPU 환경에서도 안전하게 열리도록 map_location 설정
#     model.load_state_dict(torch.load('beef_ai_brain.pth', map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # 3. 이미지 변환 규칙
# my_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# class_names = {0: "안심", 1: "우둔살", 2: "부채살"}

# # =====================================================================
# # [3] 웹페이지 화면 디자인 (Streamlit UI)
# # =====================================================================
# # 탭 이름과 아이콘 설정
# st.set_page_config(page_title="단백질 AI 스캐너", page_icon="🥩")

# # 메인 제목
# st.title("🥩 소고기 단백질 스캐너 AI")
# st.write("다이어트 식단 관리를 위한 인공지능! 고기 사진을 올리면 부위와 예상 단백질량을 알려줍니다.")

# # AI 모델 로드
# model = load_model()

# # 파일 업로드 위젯 만들기
# uploaded_file = st.file_uploader("소고기 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

# # 사용자가 사진을 업로드했을 때 실행될 내용
# if uploaded_file is not None:
#     # 1. 업로드된 이미지를 열고 화면에 보여주기
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption='업로드된 사진', use_column_width=True)

#     # 2. '분석 시작' 버튼 만들기
#     if st.button('분석 시작!'):
#         # 분석하는 동안 로딩 애니메이션 띄우기
#         with st.spinner('AI가 단백질을 분석하고 있습니다...'):
#             # 이미지 변환 및 박스 포장 (unsqueeze)
#             input_tensor = my_transform(image).unsqueeze(0)
            
#             # 예측 수행
#             with torch.no_grad():
#                 pred_class, pred_protein = model(input_tensor)
                
#                 _, predicted_idx = torch.max(pred_class, 1)
#                 predicted_name = class_names[predicted_idx.item()]
#                 predicted_protein_value = pred_protein.item()

#             # 3. 결과 예쁘게 출력하기 (화면을 좌우 두 칸으로 나누기)
#             st.subheader("📊 분석 결과")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.metric(label="예측 부위", value=predicted_name)
            
#             with col2:
#                 st.metric(label="100g당 예상 단백질", value=f"{predicted_protein_value:.1f} g")
            
#             # 성공 메시지
#             st.success("분석 완료! 근손실 없는 하루 되세요!")


import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# =====================================================================
# [1] 튜닝된 모델 뼈대 준비
# =====================================================================
class TunedMultiHeadResNet(nn.Module):
    def __init__(self, num_classes):
        super(TunedMultiHeadResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.class_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
        
        self.protein_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out_class = self.class_head(features)
        out_protein = self.protein_head(features)
        return out_class, out_protein

# =====================================================================
# [2] 뇌 불러오기 함수 (@st.cache_resource로 속도 최적화)
# =====================================================================
@st.cache_resource
def load_model():
    model = TunedMultiHeadResNet(num_classes=3)
    # 🎯 튜닝된 뇌 파일로 교체!
    model.load_state_dict(torch.load('beef_ai_brain_tuned.pth', map_location='cpu'))
    # 🎯 Dropout 끄고 실전 모드로!
    model.eval()
    return model

my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = {0: "안심", 1: "안창살", 2: "우둔살"}

# =====================================================================
# [3] Streamlit 화면 디자인
# =====================================================================
st.set_page_config(page_title="단백질 AI 스캐너", page_icon="🥩")

st.title("🥩 소고기 단백질 스캐너 AI (Tuned Pro Version)")
st.write("다이어트 식단 관리를 위한 인공지능! 고기 사진을 올리면 부위와 예상 단백질량을 알려줍니다.")

model = load_model()

uploaded_file = st.file_uploader("소고기 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='업로드된 사진', use_column_width=True)

    if st.button('분석 시작!'):
        with st.spinner('AI가 단백질을 분석하고 있습니다...'):
            input_tensor = my_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                pred_class, pred_protein = model(input_tensor)
                
                _, predicted_idx = torch.max(pred_class, 1)
                predicted_name = class_names[predicted_idx.item()]
                predicted_protein_value = pred_protein.item()

            st.subheader("📊 분석 결과")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="예측 부위", value=predicted_name)
            
            with col2:
                # 단백질 오차가 줄어들었으니 더 자신감 있게 출력!
                st.metric(label="100g당 예상 단백질", value=f"{predicted_protein_value:.1f} g")
            
            st.success("분석 완료! 근손실 없는 하루 되세요!")