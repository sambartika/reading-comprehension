Ask me anything
Reading Comprehension using SQUAD data
First install the python libraries mentioned in requirements.txt file
How to run:
cd squad
To download and process the data:
python code/preprocessing/squad_preprocess.py --data_dir data
To download GLoVE:
python code/preprocessing/download_wordvecs.py --download_dir data
To perform training:
python code/main.py --experiment_name=baseline --mode=train
To show 10 examples:
python code/main.py --experiment_name=baseline --mode=show_examples

