### This is my attempt at solving the problem given by aftershoot.

Current results on the 5 Profiles:

| Profile  | Samples | Parameter | MAE           | R2            |
|----------|---------|-----------|---------------|---------------|
| Profile1 | 811     | Temperature | 439.1647      | 0.77996       |
|          |         | Tint       | 6.1072        | 0.47345       |
| Profile2 | 750     | Temperature | 434.448       | 0.75240       |
|          |         | Tint       | 7.2351        | 0.39569       |
| Profile3 | 834     | Temperature | 453.3231      | 0.58416       |
|          |         | Tint       | 6.3271        | 0.50174       |
| Profile4 | 1544    | Temperature | 152.9643      | 0.70583       |
|          |         | Tint       | 1.6978        | 0.83232       |
| Profile5 | 811     | Temperature | 434.2257      | 0.79590       |
|          |         | Tint       | 6.1623        | 0.47334       |



## How to run the code:

set the paths to the image folder as well as the sliders csv in constants.py

after that, run the following scripts in this order

'''
python3 get_embeddings.py

python3 export_table.py

python3 train.py

python3 eval.py

'''

requirements :
opencv-python, torch, torchvision, pandas, numpy, scikit-learn, tqdm
