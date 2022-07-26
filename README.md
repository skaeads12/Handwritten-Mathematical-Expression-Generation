# Handwritten-Mathematical-Expression-Generation

이 저장소는 [Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts](https://arxiv.org/abs/2104.00887) 논문을 구현한 [MX-Font](https://github.com/clovaai/mxfont)를 수식 생성에 적합하게 수정하였습니다.

# 사전 준비

참고 데이터셋: [CROHME 손글씨 데이터셋](https://www.isical.ac.in/~crohme/CROHME_data.html)  
참고 폰트 사이트: [fontspace downloaded fonts](https://www.fontspace.com/)

1. 데이터셋을 다운로드한 뒤 .../data/data_png_TrainingPrinted/ 에 Printed 이미지 파일(.png)을 삽입합니다.
2. 손글씨 폰트를 다운로드한 뒤 .../data/ttfs/ 에 .ttf(or .otf) 파일을 삽입합니다.
> 유니코드 Mathematical Operators 블록을 지원하는 폰트일수록 정확도가 높습니다!  
[예상 결과]  
> <img src="https://i.esdrop.com/d/f/yeKdNoYiiU/RJWgllA1LN.png" width = "300" height = "300">  
3. .../cfgs/train.yaml, .../cfgs/default.yaml에서 파라미터를 수정하고 아래 학습 shell code를 입력하여 학습을 진행합니다.
4. 학습이 어느정도 진행되었다면 검증 쉘 코드를 입력하여 테스트를 수행합니다.  

# 사용법

학습

```shell

python train.py cfgs/train.yaml

```

검증

```shell

python eval.py cfgs/eval.yaml --weight result/checkpoints/last.pth --result_dir val_result/

```

# 결과

적합한 성능 측정 방법이 없어 정성평가로 대체하였습니다.  
평가 기준은 Style을 잘 캐치하여 원본과 같은 수식을 만들었을 경우 "완벽", Style을 잘 캐치하였지만 일부 원본과 다른 문자를 생성한 경우 "아쉬움", 사람이 알아보기 힘든 수식을 생성한 경우 "부족"으로 평가하였습니다.  

||완벽|아쉬움|부족|총합|
|---|---|---|---|---|
|수|46|14|4|64|
|비율|71.88%|21.88%|6.25%|100%|

### 예시
> 위: 원본 이미지(data/data_png_Printed/*.png), 아래: 생성 이미지
>  
__완벽__  
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/yKQGltEZIT.jpg" width="200" height="300">
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/kUmcKGecEz.jpg" width="200" height="300">
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/9vetcMp08s.jpg" width="400" height="300">  
  
__아쉬움__  
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/EsuYZWJ2jj.jpg" width="200" height="300">
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/WRDMRanAWA.jpg" width="200" height="300">
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/tHfv0ziSO7.jpg" width="200" height="300">  
  
__부족__  
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/Gq0d2lXpxw.jpg" width="200" height="300">
<img src="https://i.esdrop.com/d/t/yeKdNoYiiU/6nHZdSN18X.jpg" width="200" height="300">
