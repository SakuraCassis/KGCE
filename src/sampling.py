from utils import *

class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob) 
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []

        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)

        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

def sampling_paths(dataset, data, negative_num, numwalks, size):
    ap_adj = data['adjs'][0][0]
    pa_adj = ap_adj.T
    vp_adj = data['adjs'][1][1]
    pv_adj = vp_adj.T
    cv_adj = data['adjs'][1][2]
    vc_adj = cv_adj.T

    p_num = len(pa_adj)
    a_num = len(ap_adj)
    c_num = len(cv_adj)
    v_num = len(vc_adj)
    
    sampling_data_dir = 'gene/' + dataset + '/'

    if os.access(sampling_data_dir+"all_negative_samplings", os.F_OK):
        all_negative_samplings = load_data(sampling_data_dir, 'all_negative_samplings')
        all_neighbor_samplings = load_data(sampling_data_dir, 'all_neighbor_samplings')
    else:
        all_neighbor_samplings = []
        all_negative_samplings = []
        for i, adj in enumerate([ap_adj, pv_adj, vp_adj, pa_adj, cv_adj, vc_adj]):
            samplings = []
            n_samplings = []
            for j in range(len(adj)):
                node_weights = adj[j]
                weight_distribution = node_weights / np.sum(node_weights)
                samplings.append(AliasSampling(weight_distribution))

                n_weight_distribution = (node_weights-1) / np.sum((node_weights-1))
                n_samplings.append(AliasSampling(n_weight_distribution))
            all_neighbor_samplings.append(samplings)
            all_negative_samplings.append(n_samplings)
        dump_data(all_neighbor_samplings, sampling_data_dir, 'all_neighbor_samplings')
        dump_data(all_negative_samplings, sampling_data_dir, 'all_negative_samplings')

    u_d = [] #distinct type [[u_i, u_j, label, r]], r:relation type
    u_s = [] #same type [[u_i, u_j, label]]

    for i in range(numwalks):

        for p in range(p_num):
            if 1 <= size:
                # P-V
                v = all_neighbor_samplings[1][p].sampling() 
                u_d.append([p, v+p_num, 1, 0])
                for k in range(negative_num):
                    v_n = all_negative_samplings[1][p].sampling() 
                    u_d.append([p, v_n+p_num, -1, 0])

                # P-A
                a = all_neighbor_samplings[3][p].sampling() # pa
                u_d.append([p, a+c_num+p_num+v_num, 1, 11])
                for k in range(negative_num):
                    a_n = all_negative_samplings[3][p].sampling()
                    u_d.append([p, a_n+c_num+p_num+v_num, -1, 11])

            if 2 <= size:
                # P-V-P
                p1 = all_neighbor_samplings[2][v].sampling()
                u_s.append([p, p1, 1])
                for k in range(negative_num):
                    p_n = all_negative_samplings[2][v].sampling()
                    u_s.append([p, p_n, -1])

                
                # P-V-C 
                c = all_neighbor_samplings[5][v].sampling()
                u_d.append([p, c+v_num+p_num, 1, 9])
                for k in range(negative_num):
                    p_n = all_negative_samplings[5][v].sampling()
                    u_d.append([p, c+v_num+p_num, -1, 9])


                # P-A-P
                p1 = all_neighbor_samplings[0][a].sampling()
                u_s.append([p, p1, 1])
                for k in range(negative_num):
                    p_n = all_negative_samplings[0][a].sampling()
                    u_s.append([p, p_n, -1])

                # A-P-V
                v = all_neighbor_samplings[1][p1].sampling()
                u_d.append([a+c_num+p_num+v_num, v+p_num, 1, 6])
                for k in range(negative_num):
                    v_n = all_negative_samplings[1][p1].sampling()
                    u_d.append([a+c_num+p_num+v_num, v_n+p_num, -1, 6])

                # A-P-A
                a1 = all_neighbor_samplings[3][p1].sampling()
                u_s.append([a+c_num+p_num+v_num, a1+c_num+p_num+v_num, 1])
                for k in range(negative_num):
                    a_n = all_negative_samplings[3][p1].sampling()
                    u_s.append([a+c_num+p_num+v_num, a_n+c_num+p_num+v_num, 1])

                



    for v in range(v_num):
        if 1 <= size:
            # V-P
            p = all_neighbor_samplings[2][v].sampling()
            u_d.append([v+p_num, p, 1, 1])
            for k in range(negative_num):
                p_n = all_negative_samplings[2][v].sampling()
                u_d.append([v+p_num, p_n, -1, 1])

            # V-C
            c = all_neighbor_samplings[5][v].sampling()
            u_d.append([v+p_num, c+v_num+p_num, 1, 2])
            for k in range(negative_num):
                c_n = all_negative_samplings[5][v].sampling()
                u_d.append([v+p_num, c_n+v_num+p_num, -1, 2])


            
        if 2 <= size:
            # V P A
            a = all_neighbor_samplings[3][p].sampling()
            u_d.append([v+p_num, a+c_num+p_num+v_num, 1, 7])
            for k in range(negative_num):
                a_n = all_negative_samplings[3][p].sampling()
                u_d.append([v+p_num,a_n+c_num+p_num+v_num, -1, 7])



    for c in range(c_num):
        if 1 <= size:
            # C-V
            v = all_neighbor_samplings[4][c].sampling()
            u_d.append([c+v_num+p_num, v+p_num, 1, 3])
            for k in range(negative_num):
                v_n = all_negative_samplings[4][c].sampling()
                u_d.append([c+v_num+p_num, v+p_num, -1, 3])

    return u_s, u_d