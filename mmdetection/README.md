# mmdetection 
## 1. Custom Config File 생성 
- mmdetection/custom_configs/_base_ 폴더 내에 custom 할 config file을 “.py” 확장자로 작성한다. 
    
  ![스크린샷 2023-05-19 181403](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/113486402/85c2dd3a-80c7-4cdc-b22d-843aa799cd87)
- mmdetection/custom_configs 폴더 내에 최종적으로 사용할 custom.py 파일을 생성하고, _base_ 폴더에 생성한 custom config들의 경로를 설정해준다.  

  ![스크린샷 2023-05-19 182031](https://github.com/june95/Project_1/assets/113486402/7f63a3c4-3139-4c00-a948-bff4e5650c0b)
- 추가로 간단하게 config 파일을 custom하거나 변경할 경우, train.py 파일 내에서 config 파일 설정을 간단하게 변경할 수 있다. 

## 2. Train 및 Inference 
- mmdetection 폴더 내에 train.py 혹은 train_p.py 파일을 이용하여 train 및 inference가 동시에 가능하다. 
- train.py 파일에서 custom 한 “.py” 파일 경로를 설정해준다.  (아래 예시) 

  ![스크린샷 2023-05-19 182724](https://github.com/june95/Project_1/assets/113486402/8535fc36-e2e2-4f1c-a713-c808e59c4621)

- 추가 Wandb logger를 작성하여 팀 wandb에 실험 상황 공유가 가능하다. (아래 예시)  

  ![스크린샷 2023-05-19 183009](https://github.com/june95/Project_1/assets/113486402/ccdf38ca-2343-4ea4-903f-f97c89408252)

- train_p.py 파일을 이용하면, arg_parser를 이용하여 다양한 model config를 터미널에서 바꿔서 학습이 가능하다. (아래 예시)  

  ![스크린샷 2023-05-19 183555](https://github.com/june95/Project_1/assets/113486402/a38c0cf6-bd87-4e6e-846f-8b5dc7c638c2)

- 최종적으로 train.py 혹은 train_p.py 파일을 실행시키면, mmdetection 내의 inference.py 를 함께 실행시켜, 학습 후 최종 예측 csv 파일이 자동으로 생성된다.
