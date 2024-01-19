import argparse
from train import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp', help='数据集')
    parser.add_argument('--lr', type=float, default=1e-2, help='学习率') # -3
    parser.add_argument('--lr_emb', type=float, default=1e-2, help='学习率') # -3
    parser.add_argument('--lr_rs', type=float, default=1e-2, help='学习率') # -3
    parser.add_argument('--l2', type=float, default=1e-4, help='L2正则化')
    parser.add_argument('--emb_batch_size', type=int, default=512, help='批量大小')
    parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    parser.add_argument('--epochs', type=int, default=30, help='迭代次数')# 100
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--p', type=int, default=20, help='路径数量')
    parser.add_argument('--path_len', type=int, default=3, help='路径长度')
    parser.add_argument('--ratio', type=float, default=1, help='训练集使用比例')
    parser.add_argument('--is_topk', type=bool, default=True, help='top k')
    parser.add_argument('--topk', type=int, default=10, help='top k')
    parser.add_argument("--lambda1", type=float, default=1)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    print(args.dataset)

    train(args)
    