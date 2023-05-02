python main.py --model cnn --task train --shuffle False --save-dir ./checkpoints/cnn2 --log-file ./logging/cnn2_train.txt
sleep 5
python main.py --model cnn --task eval --shuffle False --save-dir ./checkpoints/cnn2 --log-file ./logging/cnn2_eval.txt
sleep 5