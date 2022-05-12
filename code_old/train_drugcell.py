import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as du
from torch.autograd import Variable
from util import *
from drugcell_NN import *


def train(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features):

    epoch_start_time = time.time()
    best_model = 0
    max_corr = 0

    # dcell neural network
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim,
                        drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final)

    train_feature, train_label, test_feature, test_label = train_data

    train_label_gpu = torch.autograd.Variable(train_label.to(opt.device))
    test_label_gpu = torch.autograd.Variable(test_label.to(opt.device))

    model.to(opt.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = model.cal_term_mask().to(opt.device)

    optimizer.zero_grad()

    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1
    # Here, the mask between Gene and relevant Term is initialized with Knowledge in Ontology
    # But its params remains to be trained

    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

    for epoch in range(train_epochs):
        model.train()
        train_predict = torch.zeros(0,0).to(opt.device)

        for i, (inputdata, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)

            cuda_features = torch.autograd.Variable(features.to(opt.device))
            cuda_labels = torch.autograd.Variable(labels.to(opt.device))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            # Here term_NN_out_map is a dictionary 
            aux_out_map, _ = model(cuda_features)

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['final'].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

            total_loss = 0    
            for name, output in aux_out_map.items():
                loss = nn.MSELoss()
                if name == 'final':
                    total_loss += loss(output, cuda_labels)
                else: # change 0.2 to smaller one for big terms
                    total_loss += 0.2 * loss(output, cuda_labels)

            total_loss.backward()

            for name, param in model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                #print name, param.grad.data.size(), term_mask_map[term_name].size()
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()

        train_corr = pearson_corr(train_predict, train_label_gpu)

        #if epoch % 10 == 0:
        torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')

        #Test: random variables in training mode become static
        model.eval()
        
        test_predict = torch.zeros(0, 0).to(opt.device)

        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)
            cuda_features = Variable(features.to(opt.device))

            aux_out_map, _ = model(cuda_features)

            if test_predict.size()[0] == 0:
                test_predict = aux_out_map['final'].data
            else:
                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

        test_corr = pearson_corr(test_predict, test_label_gpu)

        epoch_end_time = time.time()
        print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\ttotal_loss\t%.6f\telapsed_time\t%s" % (epoch, opt.device, train_corr, test_corr, total_loss, epoch_end_time-epoch_start_time))
        epoch_start_time = epoch_end_time
    
        if test_corr >= max_corr:
            max_corr = test_corr
            best_model = epoch

    torch.save(model, model_save_folder + '/model_final.pt')    

    print("Best performed model (epoch)\t%d" % best_model)





# call functions
opt = args_util()
torch.set_printoptions(precision=5)

# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(opt.train, opt.val, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_features = np.genfromtxt(opt.cell_embed, delimiter=',')
drug_features = np.genfromtxt(opt.drug_embed, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.geno_hid

num_hiddens_drug = list(map(int, opt.drug_hid.split(',')))

num_hiddens_final = opt.final_hid
train(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim, opt.save_dir, opt.epoch, opt.batch, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features)

