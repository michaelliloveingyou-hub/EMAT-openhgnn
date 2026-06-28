"""
tune_ashin_rgcn.py

RGCN + ASHIN 的调参入口。

搜索空间和 objective 都在 scripts/tune_ashin_common.py。这个文件只把默认模型设成 RGCN。
Optuna 只看验证集分数；每个 trial 都通过 subprocess 跑 main.py，和正式训练入口保持一致。

例子：
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version B --n_trials 50 --gpu 0 --seed 0
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version C --n_trials 50 --gpu 0 --seed 0
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version D --n_trials 50 --gpu 0 --seed 0
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version E --n_trials 50 --gpu 0 --seed 0
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version F --n_trials 50 --gpu 0 --seed 0
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version G --n_trials 50 --gpu 0 --seed 0
python scripts/tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version H --n_trials 50 --gpu 0 --seed 0
"""

from tune_ashin_common import main


if __name__ == "__main__":
    main(default_model="RGCN")
