import argparse
from train import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate') # -3
    parser.add_argument('--lr_emb', type=float, default=1e-2, help='embedding learning rate') # -3
    parser.add_argument('--lr_rs', type=float, default=1e-2, help='recommend learning rate') # -3
    parser.add_argument('--l2', type=float, default=1e-4, help='L2 ')
    parser.add_argument('--emb_batch_size', type=int, default=512, help='embedding batch size')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')# 100
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--p', type=int, default=20, help='path number')
    parser.add_argument('--path_len', type=int, default=3, help='path length')
    parser.add_argument('--ratio', type=float, default=1, help='training set ratio')
    parser.add_argument('--is_topk', type=bool, default=True, help='top k')
    parser.add_argument('--topk', type=int, default=10, help='top k ')
    parser.add_argument("--lambda1", type=float, default=1)
    parser.add_argument("--seed", type=int, default=111,help='random seed ')
    args = parser.parse_args()
    print(args.dataset)

    train(args)
    