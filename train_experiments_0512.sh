#!/bin/bash

# 实验的不同参数
declare -a p_unif=("0.02" "0.05" "0.1" "0.5")
declare -a dates=("240512_1" "240512_2" "240512_3" "240512_4")

# 循环数组，逐个运行实验
for i in ${!p_unif[@]}; do
  # 构建运行命令
  COMMAND="python script2.py --date ${dates[$i]} --cond_mode AdaGN --p_unif ${p_unif[$i]} --class_type label --lmda 1 --regress_type digit"

  # 在当前tmux窗口运行命令
  echo "Running command: $COMMAND"
  eval $COMMAND

  # 这里假设每个实验在完成后会自动结束
  # 如果实验会自动结束，不需要额外的等待时间或检查
  # 如果你需要确认一个实验已经结束再运行下一个，需要加入额外的逻辑来检查进程是否还在运行
done

echo "All experiments completed."
