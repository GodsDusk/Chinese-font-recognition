# Chinese-font-recognition
Chinese font recognition

## Training

- Add font files to dataset/fonts folder. Take Mac for example, those files are in /System/Library/Fonts.

- `cd dataset`

- Run `python generator.py` to generate datasets

- `cd ..`

- Run `python train.py` to train VGG16