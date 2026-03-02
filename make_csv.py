import os
import csv
from PIL import Image

# 1. 설정
image_folder = './images'
csv_filename = 'beef_dataset.csv'

# 2. 이미지 파일 목록 가져오기 (확장자 상관없이 다 가져옵니다)
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

# 3. CSV 파일 쓰기 모드로 열기
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['file_name', 'cut_class', 'protein_g']) # 첫 줄(헤더) 작성

    print("=== 미니 라벨링 툴을 시작합니다 ===")
    print("부위 코드: [0] 안심 (26g)  [1] 안창살 (22.5g)  [2] 우둔살 (19g)")
    print("종료하려면 부위 코드에 'q'를 입력하세요.\n")

    # 4. 사진을 하나씩 띄우고 질문하기
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        
        # 사진 화면에 띄우기
        img = Image.open(img_path)
        img.show() 

        # 콘솔 창에서 사용자 입력 받기
        cut_class = input(f"[{img_name}] 이 고기의 부위 코드는 무엇인가요? (0/1/2): ")
        
        if cut_class.lower() == 'q':
            print("라벨링을 중단하고 저장합니다.")
            break
            
        protein_g = input(f"[{img_name}] 이 고기의 단백질 함량(g)은 얼마인가요?: ")

        # CSV 파일에 한 줄씩 기록하기
        writer.writerow([img_name, cut_class, protein_g])
        print(f"-> 저장 완료: {img_name}, {cut_class}, {protein_g}\n")

print(f"모든 작업이 끝났습니다. '{csv_filename}' 파일이 생성되었습니다!")