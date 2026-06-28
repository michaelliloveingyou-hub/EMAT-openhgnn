# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn

"""
main.py

命令行入口。

这里负责收模型、数据集、任务、GPU、best config 等参数，然后创建 Experiment。
ASHIN 相关参数也从这里进来；真正的结构特征构造在 openhgnn/ashin 和
trainerflow/base_flow.py 里完成。

ASHIN 常用参数：
    --use_ashin
    --ashin_version {B,C,D,E,F,G,H}
    --ashin_base_version {B,C}
    --ashin_dim
    --ashin_norm
    --ashin_attr_agg
    --ashin_common_op
    --ashin_common_norm
    --ashin_common_topk
    --ashin_cache_dir
    --ashin_rebuild
    --ashin_log_dir
    --run_name
    --ashin_best_config_dir
    --ashin_best_params_path

不加 --use_ashin 时，训练流程还是 OpenHGNN 原来的逻辑。
加 --use_ashin 时，这里只传参，不直接改图。
ASHIN 场景下的 --use_best_config 会再读 Optuna 产生的 best_params.json；
best_params.json 没写到的项继续沿用 config.ini 或模型默认值。

例子：
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version B
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_best_config --use_ashin --ashin_version C
"""

import argparse
import sys


from openhgnn.experiment import Experiment


def _str2bool(value):
    # 命令行里经常会传 "True"/"False" 字符串。
    # 这里集中转成 bool，避免每个模型参数单独写一遍兼容。
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='GTN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='acm4GTN', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_distributed', action='store_true', help='will use distributed training')
    # 保留 OpenHGNN 原始参数名 --use_best_config，同时兼容用户误写/习惯写法 --use__best__config。
    parser.add_argument('--use_best_config', '--use__best__config', action='store_true',
                        help='will load utils.best_config; with ASHIN enabled, also loads ASHIN Optuna best_params.json')
    parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')
    parser.add_argument('--use_database', action='store_true',help = 'use database')
    parser.add_argument('--mini_batch_flag', action='store_true', help='will train in mini_batch mode')
    parser.add_argument('--graphbolt',action='store_true',help = 'use graphbolt to access dataset')
    parser.add_argument('--seed', default=None, type=int, help='random seed; defaults to the model config seed')
    parser.add_argument(
        '--feature_mode',
        '--feature-mode',
        choices=[
            'A', 'B', 'C', 'D', 'E',
            'raw', 'emat', 'emat_3025', 'raw_emat_3025',
            'emat_svd_64', 'emat_svd_128', 'emat_svd_256',
            'raw_emat_svd_64', 'raw_emat_svd_128', 'raw_emat_svd_256',
            'emat_tfidf', 'emat_tfidf_3025', 'raw_emat_tfidf_3025',
            'emat_tfidf_svd_64', 'emat_tfidf_svd_128', 'emat_tfidf_svd_256',
            'raw_emat_tfidf_svd_64', 'raw_emat_tfidf_svd_128', 'raw_emat_tfidf_svd_256',
            'emat_sparse_encoder', 'raw_emat_sparse_encoder',
        ],
        default='raw',
        help='Lisan feature source: A=raw, B=EMat, C=EMat-SVD-128, D=EMat-TFIDF-3025, E=raw+EMat sparse encoder.',
    )
    parser.add_argument('--use_ashin', action='store_true', help='enable ASHIN structural feature augmentation')
    parser.add_argument('--ashin_version', type=str, choices=['B', 'C', 'D', 'E', 'F', 'G', 'H'], default=None,
                        help='ASHIN feature version: B/C/D/E/F/G/H')
    parser.add_argument('--ashin_base_version', type=str, choices=['B', 'C'], default='B',
                        help='base ASHIN builder for version E/F')
    parser.add_argument('--ashin_dim', type=int, default=128, help='ASHIN-C output dimension')
    parser.add_argument('--ashin_norm', type=str, choices=['none', 'log1p', 'zscore', 'log1p_zscore'],
                        default='log1p_zscore', help='ASHIN feature normalization')
    parser.add_argument('--ashin_attr_agg', type=str, choices=['mean', 'sum', 'max'], default='mean',
                        help='ASHIN-G aggregation from attribute nodes back to target nodes')
    parser.add_argument('--ashin_common_op', type=str, choices=['max', 'sum'], default='max',
                        help='ASHIN-H operator for merging target-target commonality from attribute paths')
    parser.add_argument('--ashin_common_norm', type=str, choices=['row', 'binary'], default='row',
                        help='ASHIN-H normalization for target-target commonality propagation')
    parser.add_argument('--ashin_common_topk', type=int, default=0,
                        help='ASHIN-H top-k target commonality neighbors per target node; 0 keeps all')
    parser.add_argument('--ashin_cache_dir', type=str, default='./openhgnn/output/ashin_cache')
    parser.add_argument('--ashin_rebuild', action='store_true', help='rebuild ASHIN cache')
    parser.add_argument('--ashin_log_dir', type=str, default='./openhgnn/output/ashin_logs')
    parser.add_argument('--ashin_best_config_dir', type=str, default='./openhgnn/output/optuna',
                        help='root directory for ASHIN Optuna best_params.json')
    parser.add_argument('--ashin_best_params_path', type=str, default=None,
                        help='explicit path to an ASHIN Optuna best_params.json')
    parser.add_argument('--run_name', type=str, default=None,
                        help='ASHIN run name; defaults to model_dataset_task_ashin_seed_timestamp')

    # 这些是轻量级覆盖参数，主要供 ASHIN 单次实验和 Optuna 调参脚本使用。
    # 如果命令行没有显式传入，对应值保持 None，不会覆盖 config.ini。
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_bases', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--han_num_heads', type=int, default=None)
    parser.add_argument('--edge_dim', type=int, default=None)
    parser.add_argument('--negative_slope', type=float, default=None)
    parser.add_argument('--slope', type=float, default=None)
    parser.add_argument('--residual', type=_str2bool, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--k_layer', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--edge_drop', type=float, default=None)
    parser.add_argument('--rw_len', type=int, default=None)
    parser.add_argument('--num_node_layer', type=int, default=None)
    parser.add_argument('--num_edge_layer', type=int, default=None)
    parser.add_argument('--fanout', type=int, default=None)
    parser.add_argument('--norm', type=_str2bool, default=None)
    args = parser.parse_args()
    if args.use_ashin and args.ashin_version is None:
        parser.error('--use_ashin requires --ashin_version {B,C,D,E,F,G,H}.')

    # 记录是否“显式”写了 ASHIN 参数。
    # 这样在 --use_best_config 场景下，可以区分：
    # 1. 没写，应该使用 best_params.json 中的值；
    # 2. 写了，应该尊重命令行覆盖。
    ashin_norm_user_set = any(arg == '--ashin_norm' or arg.startswith('--ashin_norm=') for arg in sys.argv[1:])
    ashin_dim_user_set = any(arg == '--ashin_dim' or arg.startswith('--ashin_dim=') for arg in sys.argv[1:])
    ashin_base_version_user_set = any(
        arg == '--ashin_base_version' or arg.startswith('--ashin_base_version=') for arg in sys.argv[1:]
    )
    ashin_attr_agg_user_set = any(arg == '--ashin_attr_agg' or arg.startswith('--ashin_attr_agg=') for arg in sys.argv[1:])
    ashin_common_op_user_set = any(arg == '--ashin_common_op' or arg.startswith('--ashin_common_op=') for arg in sys.argv[1:])
    ashin_common_norm_user_set = any(arg == '--ashin_common_norm' or arg.startswith('--ashin_common_norm=') for arg in sys.argv[1:])
    ashin_common_topk_user_set = any(arg == '--ashin_common_topk' or arg.startswith('--ashin_common_topk=') for arg in sys.argv[1:])
    # kwargs 会传给 Experiment.set_params，成为训练流程中的 args.xxx。
    kwargs = {
        'use_distributed': args.use_distributed,
        'mini_batch_flag': args.mini_batch_flag,
        'graphbolt': args.graphbolt,
        'use_ashin': args.use_ashin,
        'ashin_version': args.ashin_version,
        'ashin_base_version': args.ashin_base_version,
        'ashin_base_version_user_set': ashin_base_version_user_set,
        'ashin_dim': args.ashin_dim,
        'ashin_dim_user_set': ashin_dim_user_set,
        'ashin_norm': args.ashin_norm,
        'ashin_norm_user_set': ashin_norm_user_set,
        'ashin_attr_agg': args.ashin_attr_agg,
        'ashin_attr_agg_user_set': ashin_attr_agg_user_set,
        'ashin_common_op': args.ashin_common_op,
        'ashin_common_op_user_set': ashin_common_op_user_set,
        'ashin_common_norm': args.ashin_common_norm,
        'ashin_common_norm_user_set': ashin_common_norm_user_set,
        'ashin_common_topk': args.ashin_common_topk,
        'ashin_common_topk_user_set': ashin_common_topk_user_set,
        'ashin_cache_dir': args.ashin_cache_dir,
        'ashin_rebuild': args.ashin_rebuild,
        'ashin_log_dir': args.ashin_log_dir,
        'ashin_best_config_dir': args.ashin_best_config_dir,
        'ashin_best_params_path': args.ashin_best_params_path,
        'run_name': args.run_name,
        'feature_mode': args.feature_mode,
    }
    # 只有命令行显式传入的参数才放进 kwargs，避免用 None 覆盖 config.ini。
    for key in ['seed', 'lr', 'weight_decay', 'hidden_dim', 'num_layers', 'patience',
                'max_epoch', 'batch_size', 'n_bases', 'num_heads', 'edge_dim', 'residual', 'beta',
                'k_layer', 'alpha', 'edge_drop', 'rw_len', 'num_node_layer', 'num_edge_layer',
                'fanout', 'norm']:
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    if args.han_num_heads is not None:
        kwargs['num_heads'] = [args.han_num_heads]
    if args.dropout is not None:
        kwargs['dropout'] = args.dropout
        if args.model == 'SimpleHGN':
            kwargs['feats_drop_rate'] = args.dropout
    if args.negative_slope is not None:
        kwargs['slope'] = args.negative_slope
    elif args.slope is not None:
        kwargs['slope'] = args.slope

    experiment = Experiment(model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu,
                            use_best_config=args.use_best_config, load_from_pretrained=args.load_from_pretrained,
                            use_database=args.use_database, **kwargs)

    experiment.run()
