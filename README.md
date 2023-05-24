![hype-squad-high-resolution-logo-color-on-transparent-background](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/72c37292-a721-4f5a-b88a-ff81d1a8ceca)
## 🚮 **재활용 품목 분류를 위한 Object Detection**
---
![comp](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/4bf8e16e-ed36-4899-996d-18273a7d8e94)
### **📆** 대회 일정 : 2023.05.03 ~ 2023.05.18

### **🗂️** Dataset

---

- 전체 이미지 개수 : 9754장 (train 4883 + test 4871 장)
- 분류 class(10개) : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- annotation file : image 정보 (id, height, width, file name) + annotation 정보 (id, Bbox, area, category id, image id)

### 📍 **프로젝트 구현 내용**

---

- **Input :** 쓰레기 객체가 담긴 이미지, Bbox (좌표, 카테고리) annotation file (COCO format)
- **Output :** Bbox 좌표, 카테고리, score 값 (Pascal VOC format)
- **Evaluation** : Test set의 mAP50(Mean Average Precision)로 평가
    - Object Detection에서 사용하는 대표적인 성능 측정 방법
    - Ground Truth 박스와 Prediction 박스 간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단

### 👨🏻‍💻 👩🏻‍💻 팀 구성  
-------------
|![logo1](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/32513a12-ac63-4d1b-9ae4-dc2dc6a7a496)|![logo2](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/8015efa2-a543-4648-aa8a-c324b6bed859)|![logo3](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/4805a650-4602-4863-98bf-b628a6085a3f)|![logo4](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/181ca1fe-022c-4641-968a-e38c496d924c)|![logo5](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/08a6781f-02ad-4f86-985d-63bc10ef3a4b)|
| :---: | :---: | :---: | :---: |  :---: |
| [김용우](https://github.com/yongwookim1) | [박종서](https://github.com/justinpark820) | [서영덕](https://github.com/SeoYoungDeok) |[신현준](https://github.com/june95) |[조수혜](https://github.com/suhyehye) |  

## 📊 EDA 결과

---

- Bbox 는 Medium, 작은 Large가 가장 많음
- Bbox 크기에 대한 class 별 imbalance는 존재하지 않았음
- 각 class 사이의 개수는 imbalance가 존재
- Train Set의 annotation을 이용하여 직접 labeling을 확인한 결과 대부분의 Bbox가 규칙성이 존재하지 않았고, 잘못 labeling되어있는 것이 상당히 많이 존재(실제 이미지는 양이 많아 첨부 생략)
- relabeling의 필요성 생각

## 🍀 Folder Structer  
``` bash
├── codebook : EDA, ensemble, visualize등의 코드를 작성  
│   ├── EDA.ipynb  
│   ├── ensemble_WBF.py  
│   ├── groupKfold.ipynb  
│   ├── pseudo_labeling.ipynb   
│   └── pyproject.toml  
├── mmdetection : mmdetection library baseline code  
│   ├── configs   
│   ├── custom_configs  
│   ├── train.py  
│   ├── train_p.py   
│   └── pyproject.toml  
└── .gitignore  
```

# mmdetection

- mmdetection 폴더 내의 README.md 참고

## 📕 Code book

---

- EDA.ipynb : train dataset EDA code
- ensemble_WBF.py : ensemble-boxes를 활용한 object detection ensemble code
- groupKfold.ipynb : train dataset을 stratified-groupKfold로 나누기 위해 사용한 code
- pseudo_labeling.ipynb : test dataset을 0.6 mAP의 모델로 pseudo labeling 하기 위한 code
- visualize_test_image.ipynb : submission file visualize code

## 💫 Final Model

---

- softNMS_WBF_1 : 모델 7개 (mAP : 0.6610 → 0.6420)
- softNMS_WBF_2 : 모델 4개 (mAP : 0.6604 → 0.6400)
- 개별 모델 중 최고 점수(mAP : 0.6283 → 0.6128, Cascade faster rcnn(backbone : swin transformer base))
- 같은 모델에서는 soft-nms 방식이 결과가 좋았고, 서로 다른 모델은 WBF 방식의 성능이 좋게 나타남
- 각각의 모델에 대해서 soft-nms를 적용한 이유는 같은 곳에 박스를 여러 번 치는 현상이 나타나 영향력을 줄이기 위해서 진행 
![final_model](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/924ca3c6-29e2-41e3-b9dc-5bb10dd3c795)

## 🔍 Reference 및 출처

---

- dataset : “부스트캠프 AI Tech”
- mmdetection

  - https://github.com/open-mmlab/mmdetection 

  - https://mmdetection.readthedocs.io/en/latest/

- UniverseNet

  - https://github.com/shinya7y/UniverseNet

- ensemble-boxes

  - https://github.com/ZFTurbo/Weighted-Boxes-Fusion 


## 📈 score graph & result 
<img width="1121" alt="ranking" src="https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/1d3b91e5-e2de-4ab7-87ad-071077569e5c">
<img width="1121" alt="ranking" src="https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113939970/74397a27-c53d-48d0-8413-0b01b6851d2d">
