from sampling import *
from model import *
import random
import torch.optim as optim

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # torch.backends.cudnn.deterministic = True

def train(args):
    # random seed
    setup_seed(args.seed)
    
    data_dir = 'data/' + args.dataset + '/'

    # load train data & pkl data：graph and HTS
    data = load_data(data_dir,"data_"+args.dataset+".pkl" )
    train_set = np.load(data_dir + str(args.ratio) + '_train_set.npy').tolist()
    test_set = np.load(data_dir + str(args.ratio) + '_test_set.npy').tolist()
    relation_dict = np.load(data_dir + str(args.ratio) + '_relation_dict.npy', allow_pickle=True).item()
    paths_dict = np.load(data_dir + str(args.ratio) + '_' + str(args.path_len) + '_path_dict.npy', allow_pickle=True).item()
    rec = np.load(data_dir + str(args.ratio) + '_rec.npy', allow_pickle=True).item()
    
    #sampling emb train data
    u_s, u_d= sampling_paths(args.dataset, data, 3, numwalks=2, size=2)
    # print (len(u_s), len(u_d))

    avg_loss = 0.
    display_batch = 1000
    batch_size = args.emb_batch_size #512
    total_batch = 10000

    test_auc_list = []
    test_acc_list = []
    test_gauc_list = []
    rec_gauc_list = []
    HR_list = []
    NDCG_list = []
    precision_list = []

    # emb train
    model = KGE(data, args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_emb, weight_decay=args.l2)
    print("begin emb train")
    
    for i in range(total_batch):
        sdx=(i*batch_size)%len(u_s)
        edx=((i+1)*batch_size)%len(u_s)

        if edx>sdx:  u_si = u_s[sdx:edx] 
        else:  u_si = u_s[sdx:]+u_s[0:edx] 
        sdx=(i*batch_size)%len(u_d)
        edx=((i+1)*batch_size)%len(u_d)
        if edx>sdx:  u_di = u_d[sdx:edx]
        else:  u_di = u_d[sdx:]+u_d[0:edx]

        model.train()
        loss,final_ent,final_rel= model(u_si, u_di)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss / display_batch
        if i % display_batch == 0 and i > 0:
            print ('%d/%d loss %8.6f' %(i,total_batch,avg_loss))
            avg_loss = 0.
    
    # RS train
    args.embeddings = final_ent
    args.relation = final_rel
    print("begin RS")
    model = PARS(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rs, weight_decay=args.l2)
    criterion = nn.BCELoss()
    
    for epoch in range(1, 1 + args.epochs):
        np.random.shuffle(train_set)
        paths, true_label, users, items = get_data(train_set, paths_dict, args.p)
        labels = torch.tensor(true_label).float().to(args.device)
        start_index = 0
        size = len(paths)
        model.train()
        while start_index < size:
            predicts = model(paths[start_index: start_index + args.batch_size],
                                relation_dict) ## forward
            
            
            loss = criterion(predicts, labels[start_index: start_index + args.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start_index += args.batch_size

        test_auc, test_acc = eval_ctr(model, test_set, paths_dict, args, relation_dict)
        test_gauc = cal_group_auc(model, test_set, paths_dict, args, relation_dict)
        rec_gauc = group_auc(model, rec, paths_dict, relation_dict, args)

        print((epoch ), '\t', test_gauc, '\t', rec_gauc, '\t', test_auc, '\t', test_acc, end='\t')
        
        HR, NDCG = 0, 0
        precisions = []
        if args.is_topk:
            HR, NDCG = eval_topk(model, rec, paths_dict, relation_dict, args.p, args.topk)
            print(HR, '\t',NDCG, end='\t')
            precisions = eval_topk1(model, rec, paths_dict, relation_dict, args.p)
            print("Precision: ", end='[')
            for i in range(len(precisions)):
                if i == len(precisions) - 1:
                    print("%.4f" % precisions[i], end=']'+'\n')
                else:
                    print("%.4f" % precisions[i], end=', ')
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        test_gauc_list.append(test_gauc )
        rec_gauc_list.append(rec_gauc)
        HR_list.append(HR)
        NDCG_list.append(NDCG)
        precision_list.append(precisions)

    print("max","auc",max(test_auc_list),"acc",max(test_acc_list),"gauc",max(test_gauc_list),\
          "HR",max(HR_list),"NDCG",max(NDCG_list),"precision_list",max(precision_list) )
    return [max(test_auc_list),max(test_acc_list),max(test_gauc_list),max(HR_list),max(NDCG_list),max(precision_list)]
    
