### This is my attempt at solving the problem given by aftershoot.

Current results on the 5 Profiles:

Profile_1
Temperature-> MAE: 594.9265393447841, R2: 0.5947148701709433
Tint-> MAE: 6.391307233147204, R2: 0.39852678779284423

Profile_2
Temperature-> MAE: 543.6956298620128, R2: 0.5865683492396532
Tint-> MAE: 7.111967769622803, R2: 0.36112784683579624

Profile_3
Temperature-> MAE: 630.4872070201122, R2: 0.3266164114253536
Tint-> MAE: 6.669532728852699, R2: 0.34049280182119224

Profile_4
Temperature-> MAE: 280.6831664443951, R2: 0.6057219909395524
Tint-> MAE: 3.170452511488156, R2: 0.6894635892239634

Profile_5
Temperature-> MAE: 609.2843635260077, R2: 0.5734372136204415
Tint-> MAE: 6.260189859258555, R2: 0.41510046177429316


How to run the code so far:

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
