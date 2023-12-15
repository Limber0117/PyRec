import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from loader_base import DataLoaderBase


class DataLoaderPYREC(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        
        
        #this is used to load the KG files, it uses knowledge graph in the model if parameter self.knowledgegraph is 1
        if self.knowledgegraph:
            kg_data = self.load_kg(self.kg_file)
        else:
            kg_data = self.load_kg(self.kg_empty) #do not use any KG information (context information)
            
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        #three columns in KG are h,r,t respectively.
        # r stands for different relationship IDs, however, the values of such relations are always 1 for project-library interactions. It is other values for context information.
        
        #there are two steps: 1) load KG and create Reverse KG (for none-exist relations in KG to improve the effectiveness) and 2) merge project-library interactions to KG.
        
        #add inverse kg data, according to previous work        
        #n_relations stands for the total number of relations+1
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        #inverse kg['r'] is from n_relations + original ['r']
        inverse_kg_data['r'] += n_relations
        #why concat kg_data and inverse_kg_data? Maybe because the relation are indeed bidirection-linked edges but the KG dataset is single-linked edges?
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        #re-map user id, prepare to add project-library interactions to existing KG.
        kg_data['r'] += 2
        #total number of relation types + 3
        self.n_relations = max(kg_data['r']) + 1 
        print("number of types of relations (containing interactions and reverse relations) is -> " + str(self.n_relations))
        
        #the total number of entities in the KG graph
        print(" - - - - - test - - - - - -")
        print("max entity id of <h> is " + str(max(kg_data['h'])))
        print("min entity id of <h> is " + str(min(kg_data['h'])))
        print("max entity id of <t> is " + str(max(kg_data['t'])))
        print("min entity id of <t> is " + str(min(kg_data['t'])))
        self.n_entities = max(max(kg_data['h']), max(kg_data['t']))+1
        print("number of entities is -> " + str(self.n_entities))
        
        ##########################################################################################
        #in current version, the KG entities have already include the user/project entities.     #
        ##########################################################################################
        #self.n_users_entities = self.n_users + self.n_entities # may be used to contact two matrices during the trainning
        self.n_users_entities = self.n_entities # may be used to contact two matrices during the trainning

        #cf_train_data contains two columns, i.e., users' IDs and items' IDs, in the training set
        #self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_train_data = (self.cf_train_data[0].astype(np.int32), self.cf_train_data[1].astype(np.int32))
        
        #cf_test_data contains two columns, i.e., users and items in the testing set
        #self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))
        self.cf_test_data = (self.cf_test_data[0].astype(np.int32), self.cf_test_data[1].astype(np.int32))

        #the train_user_dict is a dictionary, its key is the user id, and its value is an array contains all those items relavent to current user.
        #self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.train_user_dict = {k : np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        
        #self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.test_user_dict = {k : np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        #add project-library interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        #cf2kg_train_data['h'] is the head of an arrow and cf2kg_train_data['r'] is a rear of an arrow. (directed edge)
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        
        #treat each interaction as bi-direction connections in the KG
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1] #reverse the tail and head nodes
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        #KG only is used for training, so it does not have test_data.
        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)
            #may be the following two matrices are used to improve the efficiency during the training process.
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
        
        #torch.LongTensor means 64-bit integer (signed)
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    ##this is a two-dimension matrix, line is head and column is rear, value is the corresponding KG relation.
    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            
            #The following is to convert the interactions as well as relations into matrices for each "relation type".
            #### This conversion is because vals in sp.coo_matrix should be an array even it has only one value.
            vals = [1] * len(rows)
            #sp.coo_matrix((data,(rows,cols)),[dtype]) is to create a matrix with data is the element value, rows and cols are element row number and column number, respectively.
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            #the adjacency_dict stores only the positive relations, i.e., values>0.
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            print("---- initialize the laplacian matrix by symmetric ----")
            rowsum = np.array(adj.sum(axis=1, dtype=np.float32))

            #add the following line to deal with the case that the sum of a row is 0.
            #if it is 0, then change it to 1000.
            #.......to be added ...
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            print("---- initialize the laplacian matrix by random walk ----")
            rowsum = np.array(adj.sum(axis=1, dtype=np.float32))
            #print("------- rowsum is --> " + str(rowsum))
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            print(" - Laplacian r -> " + str(r))
            #print(" - Laplacian adj -> " + str(adj))
            #print("adj.sum(axis=1) -->" + str(adj.sum(axis=1)))
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info(' ---------- summary -----------')
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


