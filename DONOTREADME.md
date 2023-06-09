# Predict Student Performance from Game Play
The goal of this competition is to predict student performance during game-based learning in real-time.

Link to competition:
https://www.kaggle.com/competitions/predict-student-performance-from-game-play

### Setup
1. Clone repo
    - setup ssh connection to github 
    - run command "git clone git@github.com:Kaggle-Project-TUC-DS/student-performance.git"
2. Create conda environment and install requirements
    - check if the pip of your conda environment is in use, then run "pip install -r requirements.txt"
3. Only once - Get data with kaggle package
    - pip install kaggle
    - create api token at kaggle website/user settings
    - download kaggle.json if not triggered automatically
    - move json file to –/.kaggle/kaggle.json
    - navigate to ./data/ and run "kaggle competitions download -c predict-student-performance-from-game-play"
    - unzip 
    - rename "predict-student-performance-from-game-play" to "raw"
    - create "processed" and "final" folder (see structure)
4. OPTIONAL - Create IPython kernel from conda env
    - conda install -c anaconda ipykernel (possibly installed already)
    - python -m ipykernel install --user --name=<name_of_kernel>
5. OPTIONAL - install tensorflow_decision_forests to run example notebook
    - platform-specific: https://www.tensorflow.org/decision_forests/installation


### Structure
    .
    ├── data   
    │   ├── final                                       # data that we use when submitting
    │   ├── processed                                   # processed data
    │   └── raw                                         # the original data form the competition
    │       ├── jo_wilder/
    │       ├── sample_submission.csv
    │       ├── test.csv                                # test data
    │       ├── train_labels.csv
    │       └── train.csv                               # train data
    ├── models                                          # trained models 
    ├── notebooks                                       # pipelines for training/prediction using utils for clean code
    │   └── student-...-decision-forests.ipynb
    ├── utils                                           # functions for handling data and models, etc
    ├── README.md                                       # maintain frequently
    └── requirements.txt                                # pip install -r requirements.txt / pip freeze > requirements.txt

### Git
1. ($ git checkout -b <new branch name>)
2. $ git add . 
3. $ git commit -m “message about the commit"
4. $ git push 
