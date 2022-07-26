# Handwritten-Mathematical-Expression-Generation

이 저장소는 Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts 논문을 구현한 [MX-Font](https://github.com/clovaai/mxfont)을 수식 생성에 적합하게 수정하였습니다.

# 사전 준비

사용 데이터셋: [CROHME 손글씨 데이터셋](https://www.isical.ac.in/~crohme/CROHME_data.html)
사용 폰트: [fontspace downloaded fonts](https://www.fontspace.com/)

1. 데이터셋을 다운로드한 뒤 .../data/data_png_TrainingPrinted/ 에 .png 파일을 삽입합니다.
2. 손글씨 폰트를 다운로드한 뒤 .../data/ttfs/ 에 .ttf(or .otf) 파일을 삽입합니다.
| Mathematical Operators block 를 지원하는 폰트일수록 정확도가 높습니다!
3. .../cfgs/train.yaml, .../cfgs/default.yaml에서 파라미터를 수정하고 아래 학습 shell code를 입력하여 학습을 진행합니다.
4. 학습이 어느정도 진행되었다면 

# 사용법

학습

```shell

python train.py cfgs/train.yaml

```

검증

```shell

python eval.py cfgs/eval.yaml --weight result/checkpoints/last.pth --result_dir val_result/

```