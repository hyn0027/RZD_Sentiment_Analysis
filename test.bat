python main.py --model cnn --task train --save-dir ./checkpoints/cnn1 --log-file ./logging/cnn1_train.txt
python main.py --model cnn --task eval --save-dir ./checkpoints/cnn1 --log-file ./logging/cnn1_eval.txt

python main.py --model cnn --task train --no-shuffle --save-dir ./checkpoints/cnn2 --log-file ./logging/cnn2_train.txt
python main.py --model cnn --task eval --no-shuffle --save-dir ./checkpoints/cnn2 --log-file ./logging/cnn2_eval.txt

python main.py --model cnn --task train --optimizer SGD --lr 1e-2 --save-dir ./checkpoints/cnn3 --log-file ./logging/cnn3_train.txt
python main.py --model cnn --task eval --optimizer SGD --lr 1e-2 --save-dir ./checkpoints/cnn3 --log-file ./logging/cnn3_eval.txt

python main.py --model cnn --task train --dropout 0.1 --save-dir ./checkpoints/cnn4 --log-file ./logging/cnn4_train.txt
python main.py --model cnn --task eval --dropout 0.1 --save-dir ./checkpoints/cnn4 --log-file ./logging/cnn4_eval.txt

python main.py --model cnn --task train --dropout 0.2 --save-dir ./checkpoints/cnn5 --log-file ./logging/cnn5_train.txt
python main.py --model cnn --task eval --dropout 0.2 --save-dir ./checkpoints/cnn5 --log-file ./logging/cnn5_eval.txt

python main.py --model cnn --task train --dropout 0.4 --save-dir ./checkpoints/cnn6 --log-file ./logging/cnn6_train.txt
python main.py --model cnn --task eval --dropout 0.4 --save-dir ./checkpoints/cnn6 --log-file ./logging/cnn6_eval.txt

python main.py --model cnn --task train --batch-size 25 --save-dir ./checkpoints/cnn7 --log-file ./logging/cnn7_train.txt
python main.py --model cnn --task eval --batch-size 25 --save-dir ./checkpoints/cnn7 --log-file ./logging/cnn7_eval.txt

python main.py --model cnn --task train --batch-size 75 --save-dir ./checkpoints/cnn8 --log-file ./logging/cnn8_train.txt
python main.py --model cnn --task eval --batch-size 75 --save-dir ./checkpoints/cnn8 --log-file ./logging/cnn8_eval.txt

python main.py --model cnn --task train --lr 5e-4 --save-dir ./checkpoints/cnn9 --log-file ./logging/cnn9_train.txt
python main.py --model cnn --task eval --lr 5e-4 --save-dir ./checkpoints/cnn9 --log-file ./logging/cnn9_eval.txt

python main.py --model cnn --task train --lr 5e-3 --save-dir ./checkpoints/cnn10 --log-file ./logging/cnn10_train.txt
python main.py --model cnn --task eval --lr 5e-3 --save-dir ./checkpoints/cnn10 --log-file ./logging/cnn10_eval.txt

python main.py --model cnn --task train --cnn-kernel-num 75 --save-dir ./checkpoints/cnn11 --log-file ./logging/cnn11_train.txt
python main.py --model cnn --task eval --cnn-kernel-num 75 --save-dir ./checkpoints/cnn11 --log-file ./logging/cnn11_eval.txt

python main.py --model cnn --task train --cnn-kernel-num 125 --save-dir ./checkpoints/cnn12 --log-file ./logging/cnn12_train.txt
python main.py --model cnn --task eval --cnn-kernel-num 125 --save-dir ./checkpoints/cnn12 --log-file ./logging/cnn12_eval.txt





python main.py --model rnn --task train --save-dir ./checkpoints/rnn1 --log-file ./logging/rnn1_train.txt
python main.py --model rnn --task eval --save-dir ./checkpoints/rnn1 --log-file ./logging/rnn1_eval.txt

python main.py --model rnn --task train --no-shuffle --save-dir ./checkpoints/rnn2 --log-file ./logging/rnn2_train.txt
python main.py --model rnn --task eval --no-shuffle --save-dir ./checkpoints/rnn2 --log-file ./logging/rnn2_eval.txt

python main.py --model rnn --task train --optimizer SGD --lr 1e-2 --save-dir ./checkpoints/rnn3 --log-file ./logging/rnn3_train.txt
python main.py --model rnn --task eval --optimizer SGD --lr 1e-2 --save-dir ./checkpoints/rnn3 --log-file ./logging/rnn3_eval.txt

python main.py --model rnn --task train --dropout 0.1 --save-dir ./checkpoints/rnn4 --log-file ./logging/rnn4_train.txt
python main.py --model rnn --task eval --dropout 0.1 --save-dir ./checkpoints/rnn4 --log-file ./logging/rnn4_eval.txt

python main.py --model rnn --task train --dropout 0.2 --save-dir ./checkpoints/rnn5 --log-file ./logging/rnn5_train.txt
python main.py --model rnn --task eval --dropout 0.2 --save-dir ./checkpoints/rnn5 --log-file ./logging/rnn5_eval.txt

python main.py --model rnn --task train --dropout 0.4 --save-dir ./checkpoints/rnn6 --log-file ./logging/rnn6_train.txt
python main.py --model rnn --task eval --dropout 0.4 --save-dir ./checkpoints/rnn6 --log-file ./logging/rnn6_eval.txt

python main.py --model rnn --task train --batch-size 25 --save-dir ./checkpoints/rnn7 --log-file ./logging/rnn7_train.txt
python main.py --model rnn --task eval --batch-size 25 --save-dir ./checkpoints/rnn7 --log-file ./logging/rnn7_eval.txt

python main.py --model rnn --task train --batch-size 75 --save-dir ./checkpoints/rnn8 --log-file ./logging/rnn8_train.txt
python main.py --model rnn --task eval --batch-size 75 --save-dir ./checkpoints/rnn8 --log-file ./logging/rnn8_eval.txt

python main.py --model rnn --task train --lr 5e-4 --save-dir ./checkpoints/rnn9 --log-file ./logging/rnn9_train.txt
python main.py --model rnn --task eval --lr 5e-4 --save-dir ./checkpoints/rnn9 --log-file ./logging/rnn9_eval.txt

python main.py --model rnn --task train --lr 5e-3 --save-dir ./checkpoints/rnn10 --log-file ./logging/rnn10_train.txt
python main.py --model rnn --task eval --lr 5e-3 --save-dir ./checkpoints/rnn10 --log-file ./logging/rnn10_eval.txt

python main.py --model rnn --task train --rnn-hdim 50 --save-dir ./checkpoints/rnn11 --log-file ./logging/rnn11_train.txt
python main.py --model rnn --task eval --rnn-hdim 50 --save-dir ./checkpoints/rnn11 --log-file ./logging/rnn11_eval.txt

python main.py --model rnn --task train --rnn-hdim 70 --save-dir ./checkpoints/rnn12 --log-file ./logging/rnn12_train.txt
python main.py --model rnn --task eval --rnn-hdim 70 --save-dir ./checkpoints/rnn12 --log-file ./logging/rnn12_eval.txt

python main.py --model mlp --task train --save-dir ./checkpoints/mlp --log-file ./logging/mlp_train.txt
python main.py --model mlp --task eval --save-dir ./checkpoints/mlp --log-file ./logging/mlp_eval.txt