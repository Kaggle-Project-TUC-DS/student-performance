# Predict Student Performance from Game Play
The goal of this competition is to predict student performance during game-based learning in real-time.

Link to competition:
https://www.kaggle.com/competitions/predict-student-performance-from-game-play

## Setup
1. Clone repo
    - setup ssh connection to github 
    - run command "git clone git@github.com:Kaggle-Project-TUC-DS/student-performance.git"
2. Create conda environment and activate
    - commands
3. OPTIONAL - Kaggle package (helpful for downloading data and might be useful for submissions)
    - pip install kaggle
    - create api token at kaggle website/user settings
     - download kaggle.json if not triggered automatically
    - move json file to –/.kaggle/kaggle.json
    - navigate to ./data/ folder and run "kaggle competitions download -c predict-student-performance-from-game-play"
    - unzip 
    - move data up so it fits the structure below

## Structure
    .
    ├── data                        # data files
    │   ├── jo_wilder/
    │   ├── sample_submission.csv
    │   ├── test.csv                # test data
    │   ├── train_labels.csv
    │   └── train.csv               # train data
    ├── utils                       # utilities
    └── README.md

