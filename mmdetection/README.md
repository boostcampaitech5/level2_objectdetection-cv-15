# mmdetection 
## 1. Custom Config File 생성 
- mmdetection/custom_configs/_base_ 폴더 내에 custom 할 config file을 “.py” 확장자로 작성한다. 
    
  ![스크린샷 2023-05-19 181403](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113486402/16671b20-f420-4afc-9a62-dbb19976a243)

- mmdetection/custom_configs 폴더 내에 최종적으로 사용할 custom.py 파일을 생성하고, _base_ 폴더에 생성한 custom config들의 경로를 설정해준다.  

  ![스크린샷 2023-05-19 182031](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113486402/56e09dc2-6e4f-4e73-a908-be45412858c3)

- 추가로 간단하게 config 파일을 custom하거나 변경할 경우, train.py 파일 내에서 config 파일 설정을 간단하게 변경할 수 있다. 

## 2. Train 및 Inference 
- mmdetection 폴더 내에 train.py 혹은 train_p.py 파일을 이용하여 train 및 inference가 동시에 가능하다. 
- train.py 파일에서 custom 한 “.py” 파일 경로를 설정해준다.  (아래 예시) 

  ![스크린샷 2023-05-19 182724](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113486402/3cb71d00-e4dc-4a8f-86b9-be5adad80993)


- 추가 Wandb logger를 작성하여 팀 wandb에 실험 상황 공유가 가능하다. (아래 예시)  

  ![스크린샷 2023-05-19 183009](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113486402/93cfa976-2a45-47de-85eb-396ac25d79b8)


- train_p.py 파일을 이용하면, arg_parser를 이용하여 다양한 model config를 터미널에서 바꿔서 학습이 가능하다. (아래 예시)  

  ![스크린샷 2023-05-19 183555](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113486402/a4baa861-bb5f-474d-9d5a-6ce40323f18e)


- 최종적으로 train.py 혹은 train_p.py 파일을 실행시키면, mmdetection 내의 inference.py 를 함께 실행시켜, 학습 후 최종 예측 csv 파일이 자동으로 생성된다.
