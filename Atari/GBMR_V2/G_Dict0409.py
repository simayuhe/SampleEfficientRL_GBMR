
'''
#0330
最简版本的图

#0331
由于每个动作一个图会存在单结点存入的时候对虚拟边冗余的情况，所以把整体的索引结构改成一个图

不在区分状态是在哪个边上的

所以需要：
1. 改进annoy list的读写
2. 改进图的结构
3. 改进图的读写操作
4. 重新构造网络的训练方法（应用到带编码器的结构中时）

先测试 add pair
再进行批量写入，批量写入主要是针对 n-step的时候要根据一条轨迹算回报
query 的时候 可以把多个动作的Q值都放到这里，但是这样进行训练的时候拿到的就是所有的，可能还要再分离出去

# 0407
引入轨迹的时候要把最末尾的状态加入，这样可以得到完整的图，此时的features比 actions 多了一个，因为最后一个状态上没有做动作

#0407 
重构的时候 returns 用的是R，而不是Q

'''


import numpy as np
from annoy import AnnoyIndex
import networkx as nx
#from CanopyForNodeAttributes import Canopy as Cluster
import time
import multiprocessing

class Graph_dict():
    def __init__(self,capacity,key_dimension,act_dimension,dist_th,eta=0.01,batch_size=64):
        # 一个图带19个表，每个表是一种动作的特征集合
        # 一个图只带一个表，表中只存储现有节点
        self.capacity = capacity # 每个表的容量
        self.key_dimension = key_dimension
        self.act_dimension = act_dimension
        self.G = nx.DiGraph()
        self.node_features_list = np.zeros((capacity,key_dimension))
        self.indices_list = AnnoyIndex(key_dimension,metric='euclidean')
     
        self.initial_update_size = batch_size
        self.min_update_size = self.initial_update_size # 隔一段时间更新一次
        self.cached_keys = []
        self.cached_indices =[]

        self.lru = np.zeros(self.capacity)
        self.tm = 0.0
        self.curr_capacity =0
        self.built_capacity =0

        self.eta = eta # 控制边权的增长速度
        self.dist_th = dist_th
        self.gamma =0.9


    def Graphisfull(self):
        if self.built_capacity<self.capacity:
            return False
        return True
 
    def queryable(self,k):
        if self.curr_capacity>k and self.built_capacity>k:
            return True
        else:
            return False    

    def _insert(self, keys, indices):
        self.cached_keys = self.cached_keys + keys
        self.cached_indices = self.cached_indices + indices
        
        if len(self.cached_indices) >= self.min_update_size:# 为啥设置这个阈值
            self.min_update_size = max(self.initial_update_size, self.curr_capacity*0.02)
            self._update_index()
 
    def _update_index(self):
        self.indices_list.unbuild()
        for i, ind in enumerate(self.cached_indices):
            new_key = self.cached_keys[i]
            self.node_features_list[ind,:] = new_key
            self.indices_list.add_item(ind,new_key)
        self.cached_keys = []
        self.cached_indices = []
        self.indices_list.build(50)
        self.built_capacity = self.curr_capacity

    def _refresh(self, keys, indices):
        pass
        # self.indices_list = AnnoyIndex(key_dimension,metric='euclidean')
        # # 重建一个list
        # # 找到所有的节点特征
        # # 用节点特征和编号重新构图

    def check_state_index(self,s):
        if self.queryable(1):
            ind,dist =self.indices_list.get_nns_by_vector(s,1,include_distances=True)# 找到最近的一个状态
            d = dist[0] # 读取它 的距离
            if d <self.dist_th:# 对于确定环境可以把这个值限定的比较小
                #print("如果存在非常相似的，就直接对其边权进行更新就行了")
                temp_index_s = ind[0]
                key_s = (self.indices_list.get_item_vector(temp_index_s) + s*d)/(1+d)
                # 对节点特征list 不进行更新，
            elif self.curr_capacity >= self.capacity:
                #print("# 如果不是很近，但是图已经满了")
                temp_index_s =np.argmin(self.lru)
                self.G.remove_node(temp_index_s)
                #self._refresh([key_s], [temp_index_s])
                key_s = s
                # 在索引表中替换掉原先的
            else:
                #print(" # 不是很像，而且图没满")
                temp_index_s = self.curr_capacity
                self.curr_capacity += 1
                key_s =s
                # 在索引表中新加一个

        else:
            #print("# 之前是个空的表")
            temp_index_s = self.curr_capacity
            key_s =s 
            self.curr_capacity += 1
            # 在索引表中新加一个
        self._insert([key_s], [temp_index_s])
        self.tm += 0.01   
          
        self.lru[temp_index_s] = self.tm
        return temp_index_s

    def add_by_pair(self,s,a,r,s_):

        # 这个比Q表多了一个定位过程,每次写入一条边
        start_node=self.check_state_index(s) # 通过观测查状态
        if start_node in self.G:
            pass
        else:
            self.G.add_node(start_node)
        end_node = self.check_state_index(s_)
        if start_node==end_node:
            return True
        if end_node in self.G: # 如果下一个节点存在
            if self.G.has_edge(start_node, end_node):# 看是否有边
                old_weight = self.G[start_node][end_node]["weight"]
                q_next = 0
                for edge_i in self.G.out_edges(end_node):
                    if len(edge_i) == 0:
                        continue
                    if q_next<self.G[edge_i[0]][edge_i[1]]["weight"]:
                        q_next = self.G[edge_i[0]][edge_i[1]]["weight"]
                target_weight = r + self.gamma*q_next 
                new_weight = old_weight + self.eta*(target_weight-old_weight)
                self.G.add_edge(start_node,end_node,label=a,weight=new_weight,reward=r)
            else:
                self.G.add_edge(start_node,end_node,label=a,weight=r,reward=r)
        else:# 如果下一个节点不存在
            self.G.add_node(end_node)
            self.G.add_edge(start_node, end_node,label=a,weight=r,reward=r)

    def add_by_trajectory(self,features,actions,returns):
        len_features =len(features)
        encoded_trj =  []
        for i in range(len_features-1,0,-1):
            s = features[i]
            if i ==len_features-1:
                # 对于轨迹中的最后一个点，这个点上没有动作，只是作为终点放在这儿
                end_node = self.check_state_index(s)
                encoded_trj.append(end_node) 
                if end_node in self.G:
                    pass
                else:
                    self.G.add_node(end_node)
                continue
            else:
                # 其它的点
                # time_check_a = time.time()
                start_node = self.check_state_index(s)
                # time_check_b = time.time()
                # print("check time",time_check_b-time_check_a)
                encoded_trj.append(start_node) 
                # time_addedge_a = time.time()
                if start_node == end_node:
                    continue
                if start_node in self.G:
                    pass
                else:
                    self.G.add_node(start_node)
                if self.G.has_edge(start_node, end_node):# 看是否有边
                    old_weight = self.G[start_node][end_node]["weight"]
                    q_next = 0
                    for edge_i in self.G.out_edges(end_node):
                        if len(edge_i) == 0:
                            continue
                        if q_next<self.G[edge_i[0]][edge_i[1]]["weight"]:
                            q_next = self.G[edge_i[0]][edge_i[1]]["weight"]
                    target_weight = returns[i] + self.gamma*q_next 
                    new_weight = old_weight + self.eta*(target_weight-old_weight)
                    self.G.add_edge(start_node,end_node,label=actions[i],weight=new_weight,reward=returns[i])
                else:
                    self.G.add_edge(start_node,end_node,label=actions[i],weight=returns[i],reward=returns[i])
                # time_addedge_b = time.time()
                # print("add edge time ",time_addedge_b-time_addedge_a)
                end_node = start_node # 这个不能忘了
        return encoded_trj

    def update_by_indices(self,nodes,actions,returns):
        # 因为已经知道点的标号了，也就是我们并不准备在这个函数中改变图的拓扑结构，只更新边权
        len_features =len(nodes)
        for i in range(len_features-1,0,-1):
            if i == len_features -1 :
                end_node= nodes[i]
                continue
            else:
                start_node = nodes[i]
                if start_node ==end_node:
                    continue
                else:
                    old_weight = self.G[start_node][end_node]["weight"]
                    q_next = 0
                    for edge_i in self.G.out_edges(end_node):
                        if len(edge_i)==0:
                            continue
                        if q_next<self.G[edge_i[0]][edge_i[1]]["weight"]:
                            q_next = self.G[edge_i[0]][edge_i[1]]["weight"]
                    target_weight = returns[i] + self.gamma*q_next # 这里的values 是r 或者nstep r
                    new_weight = old_weight + self.eta*(target_weight-old_weight)
                    self.G.add_edge(start_node,end_node,label=actions[i],weight=new_weight,reward=returns[i])
                    self.lru[start_node] =self.tm

                end_node = start_node
        self.tm +=0.01        
        
    def query_by_features(self,features,actions,k):
        # 根据一串n 个 点的特征和动作，求多个近邻，及相应的Q值，仍然是对动作查的，并不是整个读取
        Q_s = np.zeros((len(features),k)) # n *k 
        Embs = np.zeros((len(features),k,self.key_dimension))# n*k* key_dim
        for i,f in enumerate(features):
            a = actions[i]
            inds,dists = self.indices_list.get_nns_by_vector(f,k,include_distances= True)
            for j,ind in enumerate(inds):
                # if dists[j]>self.dist_th:# 距离太远了就不要了，Q值默认是0，Emb .默认也是【0 0 0 0】
                #     continue # 这和上面求近邻的不能是统一个参数，否则的话就是小于T换掉，大于T不用，就剩自己了
                index = ind 
                for edge_i in self.G.out_edges(index):
                    if len(edge_i)==0:# 边是空的就不管了
                        continue
                    if self.G[edge_i[0]][edge_i[1]]["label"] == a: # 如果边的类型是a ，我们读取相应的Q值
                        Q_s[i,j] = self.G[edge_i[0]][edge_i[1]]["weight"] 
            Embs[i,:,:] = self.node_features_list[inds,:]
            self.lru[inds] = self.tm 
        self.tm += 0.01

        return Q_s,Embs   

    def GetKeyPointByDegree(self,num_center):
        # # 每个节点的出度，入度 加和，并除以 访问次数开根号
        # # 这只是众多可以找到关键节点的方法中的一种
        # # 先看看大家的出度入度都是什么量级的
        # print(nx.degree_histogram(self.G)) # [0, 2, 587, 31, 55, 18, 29, 15, 13, 12, 7, 3, 3, 2, 1, 1, 2, 0, 1] 是个统计图，表示出度为18的点有1个
        # # 我们可以对这个分布截尾10%，然后最低的那个作为阈值，得到相应的点，这些点的个数应该不会多
        # degree_th = len(nx.degree_histogram(self.G))*0.7
        # print(degree_th) 
        # # 把这些点的频次拿出来，除，再取前50% 然后得到的就是出现次数不是很多，但是节点连接丰富的地方
        # # 重构的时候想办法按奖励找路径，同时还要考量一下中间节点个数不能太多
        N = np.sum(self.built_capacity)
        self.SI = np.zeros(self.capacity*self.act_dimension)
        #self.SU = np.zeros(self.capacity*self.act_dimension)
        for node in self.G.nodes():
            fb = len(self.G.in_edges(node)) # before
            fa = len(self.G.out_edges(node)) # after
            #ns = self.G.nodes[node]["visit"]
            #si = (fa+fb)/(np.sqrt(ns/N+1e-6))
            si = (fa+fb)
            #print("fb",fb,"fa",fa,"ns",ns,"ns/N",ns/N,"fa+fb",fa+fb,"Si",si)
            self.SI[node]= si
            #self.SU[node]= fb+fa
        Sorted_si_ind = np.argsort(self.SI,axis=0)[-num_center:-1]
        # Sorted_si = self.SI[Sorted_si_ind]
        # Sorted_su_ind = np.argsort(self.SU,axis=0)[-20:-1]
        # Sorted_su = self.SU[Sorted_su_ind]
        # print("S si ind",Sorted_si_ind,"s si",Sorted_si,"\n","s su ind",Sorted_su_ind,"s su",Sorted_su)
        center_list=list(Sorted_si_ind)
        return center_list
    
    def FindValuablePathbyLength(self,start_node,end_node):
        path =[]
        temp_actions = []
        temp_rewards = []
        if nx.algorithms.shortest_paths.generic.has_path(self.G,start_node,end_node):
            path = nx.shortest_path(self.G, start_node,end_node)                              
            temp_actions = []
            temp_weights = []
            for idx in range((len(path)-1)):
                pair_start = path[idx]
                pair_end = path[idx+1]
                temp_actions.append(self.G.edges[pair_start,pair_end]['label'])
                temp_rewards.append(self.G.edges[pair_start,pair_end]['reward'])# 这里不应该加入 当前的q值，而是用R
            #print(len(path),len(temp_actions),len(temp_weights)) # path会比action多一个，最后一个没算进去
        return path, temp_actions,temp_rewards

    def FindValuablePathbyReturns(self,start_node,end_node):
        path = []
        pass


# 用来测试这个类的
if __name__ == '__main__':
    # 用来对类内新增函数作简单测试
    G= Graph_dict(70,8,9) # 7个数据，每个数据的特征是8维，有9个动作
    # print("self.adj_matrix",G.adj_matrix.shape)
    # print("self.weight_matrix",G.weight_matrix.shape)
    # print("self.node_features ",G.node_features.shape)
    # print("self.node_attributes ",G.node_attributes.shape)

    embs = []
    acts = []
    rews = []
    for i in range(40):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    #print(embs,acts,rews)

    G.add_by_features(embs,acts,rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])
    embs = []
    acts = []
    rews = []
    for i in range(50):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    #print(embs,acts,rews)

    G.add_by_features(embs,acts,rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])
    embs = []
    acts = []
    rews = []
    for i in range(30):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    #print(embs,acts,rews)

    G.add_by_features(embs,acts,rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])

    G.GraphCluster(5,4)