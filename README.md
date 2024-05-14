train using classified data

python script.py --date 240508_1 --cond_mode Attention --classifier_name 0 --class_type logit

date {240508_1-n}
cond_mode {Attention, AdaGN}
classifier_name {0,0.01,0.05,0.1,1}
class_type {logit, label}

skip 240508_1 Attention 0 logit

python script.py --date 240508_21 --cond_mode Attention --p_unif 0.01 --class_type label

python script2.py --date 240511_5 --cond_mode AdaGN --p_unif 0.01 --class_type label --lmda 0.1 --regress_type digit


python test_script.py --date 240513_1 --model_name ./experiments/240511_5/model/model_29.pth --clean_classifier_name ./trained_classifiers/model_gray_clean_60.pt --noisy_classifier_name ./trained_classifiers/model_gray_noisy_60.pt --dataset_type ID --batch_size 16 --batch_num 100 > ./test_experiments/240513_1/log/stdout.log

python test_script.py --date 240513_2 --model_name ./diff_to_download/240504_2_model_29.pth --clean_classifier_name ./trained_classifiers/model_gray_clean_60.pt --noisy_classifier_name ./trained_classifiers/model_gray_noisy_60.pt --dataset_type ID --batch_size 16 --batch_num 100 > ./test_experiments/240513_2/log/stdout.log

python test_script.py --date 240513_3 --model_name ./experiments/240511_5/model/model_29.pth --clean_classifier_name ./trained_classifiers/model_gray_clean_60.pt --noisy_classifier_name ./trained_classifiers/model_gray_noisy_60.pt --dataset_type OOD --batch_size 16 --batch_num 100 > ./test_experiments/240513_3/log/stdout.log

python test_script.py --date 240513_4 --model_name ./diff_to_download/240504_2_model_29.pth --clean_classifier_name ./trained_classifiers/model_gray_clean_60.pt --noisy_classifier_name ./trained_classifiers/model_gray_noisy_60.pt --dataset_type OOD --batch_size 16 --batch_num 100 > ./test_experiments/240513_4/log/stdout.log

python test_script.py --date 240513_5 --model_name ./experiments/240511_5/model/model_29.pth  --dataset_type ID --batch_size 32 --batch_num 100 > ./test_experiments/240513_5/log/stdout.log

python test_script.py --date 240513_6 --model_name ./diff_to_download/240504_2_model_29.pth  --dataset_type ID --batch_size 32 --batch_num 100 > ./test_experiments/240513_6/log/stdout.log

python test_script.py --date 240513_7 --model_name ./experiments/240511_5/model/model_29.pth  --dataset_type OOD --batch_size 32 --batch_num 100 > ./test_experiments/240513_7/log/stdout.log

python test_script.py --date 240513_8 --model_name ./diff_to_download/240504_2_model_29.pth  --dataset_type OOD --batch_size 32 --batch_num 100 > ./test_experiments/240513_8/log/stdout.log

python test_script.py --date 240513_9 --model_name ./experiments/240511_1/model/model_29.pth  --dataset_type ID --batch_size 32 --batch_num 100 > ./test_experiments/240513_9/log/stdout.log


python script2.py --date 240514_1 --cond_mode AdaGN --p_unif 0 --class_type label --lmda 1 --regress_type digit

python script2.py --date 240514_2 --cond_mode AdaGN --p_unif 0 --class_type label --lmda 0.1 --regress_type digit

python script_cvae.py --date 240514_3 --p_unif 0.01

python script2_cvae.py --date 240514_4 --p_unif 0.01 --lmda 10000


python script_cvae.py --date 240514_5 --p_unif 0 (on _2)

python script2_cvae.py --date 240514_6 --p_unif 0 --lmda 10000  (on _2)

以下4个 (on _1)

python test_script.py --date 240514_7 --model_name ./experiments/240514_2/model/model_29.pth  --dataset_type ID --batch_size 32 --batch_num 50 > ./test_experiments/240514_7/log/stdout.log

python test_script.py --date 240514_8 --model_name ./diff_to_download/240504_1_model_29.pth  --dataset_type ID --batch_size 32 --batch_num 50 > ./test_experiments/240514_8/log/stdout.log

python test_script.py --date 240514_9 --model_name ./experiments/240514_2/model/model_29.pth  --dataset_type OOD --batch_size 32 --batch_num 50 > ./test_experiments/240514_9/log/stdout.log

python test_script.py --date 240514_10 --model_name ./diff_to_download/240504_1_model_29.pth  --dataset_type OOD --batch_size 32 --batch_num 50 > ./test_experiments/240514_10/log/stdout.log

python script.py --date 240514_11 --cond_mode AdaGN --p_unif 0 --class_type label

python script2_cvae.py --date 240514_12 --p_unif 0 --lmda 500 

TODO: (add LPIPS; test naive p_unif=0)
(remove lpips)
(cil p_unif=0)
python test_script.py --date 240514_13 --model_name ./experiments/240514_2/model/model_29.pth  --dataset_type ID --batch_size 32 --batch_num 50 > ./test_experiments/240514_13/log/stdout.log

(naive p_unif=0)
python test_script.py --date 240514_14 --model_name ./experiments/240514_11/model/model_29.pth  --dataset_type ID --batch_size 32 --batch_num 50 > ./test_experiments/240514_14/log/stdout.log


TODO: (cosine similarity between extracted features of different color)

(cil, p_unif=0.01)
python feature_distance.py --date 250514_15 --model_name ./experiments/240511_5/model/model_29.pth
(naive, p_unif=0.01)
python feature_distance.py --date 250514_16 --model_name ./diff_to_download/240504_2_model_29.pth
(cil, p_unif=0)
python feature_distance.py --date 250514_17 --model_name ./experiments/240514_2/model/model_29.pth
(naive, p_unif=0)
python feature_distance.py --date 250514_18 --model_name ./experiments/240514_11/model/model_29.pth

(cil, p_unif=0.01, 只plot一个形状)
python feature_distance.py --date 250514_19 --model_name ./experiments/240511_5/model/model_29.pth --plot_digit 2
(naive, p_unif=0.01, 只plot一个形状)
python feature_distance.py --date 250514_20 --model_name ./diff_to_download/240504_2_model_29.pth --plot_digit 2