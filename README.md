train using classified data

python script.py --date 240508_1 --cond_mode Attention --classifier_name 0 --class_type logit

date {240508_1-n}
cond_mode {Attention, AdaGN}
classifier_name {0,0.01,0.05,0.1,1}
class_type {logit, label}

skip 240508_1 Attention 0 logit

python script.py --date 240508_21 --cond_mode Attention --p_unif 0.01 --class_type label