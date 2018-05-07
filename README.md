Ask me anything<br />
Reading Comprehension using SQUAD data<br />
First install the python libraries mentioned in requirements.txt file<br />
How to run:<br />
cd squad<br />
mkdir data<br />
To download and process the data:<br />
python code/preprocessing/squad_preprocess.py --data_dir data<br />
To download GLoVE:<br />
python code/preprocessing/download_wordvecs.py --download_dir data<br />
To perform training:<br />
python code/main.py --experiment_name=baseline --mode=train<br />
To show 10 examples:<br />
python code/main.py --experiment_name=baseline --mode=show_examples<br />

