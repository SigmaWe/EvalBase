mkdir -p cnndm
python3 -m pip install gdown 
gdown "https://drive.google.com/u/0/uc?export=download&confirm=Fiu7&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"

gdown "https://drive.google.com/u/0/uc?export=download&confirm=0051&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs"

tar -xvzf cnn_stories.tgz -C cnndm

tar -xvzf dailymail_stories.tgz -C cnndm

wget -O summeval_annotations.aligned.scored.jsonl "https://drive.google.com/u/0/uc?id=1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS&export=download"

wget https://raw.githubusercontent.com/Yale-LILY/SummEval/master/data_processing/pair_data.py

python3 pair_data.py --data_annotations summeval_annotations.aligned.scored.jsonl --story_files .

# rm -r cnndm/
