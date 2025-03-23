### This is my attempt at solving the problem given by aftershoot.

Current results on the 5 Profiles:

| Profile  | Samples | Parameter   | MAE        | R2        |
|----------|---------|-------------|------------|-----------|
| Profile1 | 811     | Temperature | 401        | 0.8314    |
|          |         | Tint        | 5.5004     | 0.5806    |
| Profile2 | 750     | Temperature | 372        | 0.7897    |
|          |         | Tint        | 6.3857     | 0.4729    |
| Profile3 | 834     | Temperature | 427        | 0.6668    |
|          |         | Tint        | 5.8797     | 0.5857    |
| Profile4 | 1544    | Temperature | 141        | 0.6581    |
|          |         | Tint        | 1.6978     | 0.83232   |
| Profile5 | 811     | Temperature | 408        | 0.8223    |
|          |         | Tint        | 5.4756     | 0.5824    |



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
