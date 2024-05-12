train using classified data

python script.py --date 240508_1 --cond_mode Attention --classifier_name 0 --class_type logit

date {240508_1-n}
cond_mode {Attention, AdaGN}
classifier_name {0,0.01,0.05,0.1,1}
class_type {logit, label}

skip 240508_1 Attention 0 logit

python script.py --date 240508_21 --cond_mode Attention --p_unif 0.01 --class_type label

python script2.py --date 240511_5 --cond_mode AdaGN --p_unif 0.01 --class_type label --lmda 0.1 --regress_type digit

python test_script.py --date 240513_1 --model_name ./experiments/240511_5/model/model_29.pth --clean_classifier_name ./trained_classifiers/model_gray_clean_40.pt --noisy_classifier_name ./trained_classifiers/model_gray_noisy_40.pt --dataset_type ID --batch_size 16 --batch_num 100