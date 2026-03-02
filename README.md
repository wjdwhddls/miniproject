# 🥩 소고기 단백질 스캐너 AI (Beef Protein Scanner)

다이어트 및 근육량 유지를 위한 맞춤형 인공지능 프로젝트입니다. 
스마트폰이나 갤러리에 있는 소고기 사진을 업로드하면, AI가 **고기의 부위를 분류**하고 **100g당 예상 단백질량을 예측**하여 웹 화면에 즉시 보여줍니다.

## 🎯 프로젝트 목표
- **개인적 동기:** 다이어트 중 근육량 유지를 위해 정확한 단백질 섭취량 계산이 필요했습니다.
- **기술적 동기:** 단순한 이미지 분류(Classification)를 넘어, 하나의 신경망 뼈대(Backbone)로 두 가지 다른 결과(분류 및 회귀)를 동시에 예측하는 **Multi-Head Architecture**를 직접 설계하고 구현해보고자 하였습니다.

## 🛠️ 기술 스택 (Tech Stack)
- **Language:** Python 3.10
- **Deep Learning Framework:** PyTorch, Torchvision
- **Web Frontend:** Streamlit
- **Data Manipulation:** Pandas, PIL (Pillow)
- **Environment:** Conda (macOS)

---

## 🚀 개발 파이프라인 및 핵심 구현 사항

### Step 1. 데이터 수집 및 전처리 (Data Preparation)
- **데이터셋 구축:** 안심(0), 안창살(1), 우둔살(2) 총 3가지 부위의 소고기 이미지를 직접 수집 (총 30장).
- **정답지(Labeling) 제작:** 파이썬 스크립트(`make_csv.py`)를 활용하여 각 이미지의 부위 클래스와 100g당 평균 단백질량(g)을 매핑한 `beef_dataset.csv` 파일 생성.
- **데이터 증강(Data Augmentation):** 데이터 부족으로 인한 과적합을 막기 위해 `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` 등을 적용하여 모델의 일반화 성능 확보.
- **Troubleshooting:** 학습 중 단백질 예측 오차가 튀는 현상을 발견, CSV 데이터를 전수조사하여 라벨링 휴먼 에러(Garbage Data)를 직접 찾아내고 수정하여 모델 안정화.

### Step 2. 객체 지향(OOP) 기반 Multi-Head 모델 설계
- `torch.nn.Module`을 상속받아 커스텀 클래스 `TunedMultiHeadResNet` 구현.
- **Backbone (공통 뼈대):** 사전 학습된 `ResNet18`을 가져와 마지막 Fully Connected Layer(`fc`)를 제거하고 특징 추출기(Feature Extractor)로 활용.
- **Multi-Head 구성:** 1. `class_head`: 3개의 부위를 맞추는 분류 머리 (`CrossEntropyLoss` 사용)
  2. `protein_head`: 단백질 수치(연속형 변수)를 맞추는 회귀 머리 (`MSELoss` 사용)

### Step 3. 모델 아키텍처 튜닝 (과적합 방지)
단순한 선형 연결(`nn.Linear`)만 있던 초기 모델에 **비선형성과 규제(Regularization)**를 추가하여 성능을 극대화했습니다.
- **`nn.ReLU()` 적용:** 복잡한 고기 마블링 패턴을 학습할 수 있도록 판단력 강화.
- **`nn.Dropout()` 적용:** 분류 헤드(p=0.5)와 회귀 헤드(p=0.3)에 모래주머니 효과를 주어, 모델이 특정 배경(접시 색깔 등)을 암기하지 않고 고기의 본질적인 특징에 집중하도록 과적합 방지.

### Step 4. 모델 학습 및 검증
- `Adam` Optimizer(학습률 0.001)를 사용하여 총 20 Epoch 동안 학습 진행.
- 훈련 모드(`model.train()`)와 평가 모드(`model.eval()`)를 엄격하게 분리하여 Dropout의 활성화 시점을 제어.
- 최종 학습된 가중치는 `beef_ai_brain_tuned.pth` 파일로 영구 저장.

### Step 5. 실전 테스트 및 웹 서비스 배포 (Streamlit)
- 무거운 학습 코드를 제외하고 가벼운 뼈대에 뇌(가중치)만 이식하는 추론 전용 스크립트 분리.
- **Streamlit**을 도입하여 사용자가 브라우저에서 직관적으로 사진을 올리고 결과를 확인할 수 있는 UI/UX 구축.
- `@st.cache_resource` 데코레이터를 사용하여 웹페이지 새로고침 시 모델을 매번 다시 불러오는 병목 현상 제거 (최적화).

---

## 📁 프로젝트 폴더 구조 (Project Structure)
```text
miniproject/
 ├── images/                 # 학습 및 테스트용 소고기 이미지 데이터
 ├── beef_dataset.csv        # 부위 및 단백질량이 기록된 정답지(Label)
 ├── train.py                # Dataset 로드 및 모델 튜닝/학습/저장 코드
 ├── predict.py              # 터미널 환경에서의 빠른 추론 테스트 코드
 ├── app.py                  # Streamlit 기반의 웹 어플리케이션 코드
 └── beef_ai_brain_tuned.pth # 학습이 완료된 딥러닝 모델 가중치 파일
