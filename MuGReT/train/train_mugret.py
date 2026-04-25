from PycharmProject.MuGReT-DWA.MuGReT.util.setting import log
from PycharmProject.MuGReT-DWA.MuGReT.nets.mugret_network import MuGReTNet
from PycharmProject.MuGReT-DWA.MuGReT.eval.load_eval import bcb_evaluation
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import time


def transfer_label(label):
    if label == 0:
        return -1
    else:
        return 1


def evaluation_mugret(model, dataset, params, net_params):
    device = net_params['device']
    bcb_samples = []
    results = []
    start_time = time.time()
    with torch.no_grad():
        for data in tqdm(dataset, desc="Processing Dataset", total=len(dataset)):
            ast_1, ast_2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2 = data.ast_1, data.ast_2, data.ast_edge_index_1, data.ast_edge_index_2, data.ast_edge_attr_1, data.ast_edge_attr_2
            cfg_1, cfg_2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2 = data.cfg_1, data.cfg_2, data.cfg_edge_index_1, data.cfg_edge_index_2, data.cfg_edge_attr_1, data.cfg_edge_attr_2
            dfg_1, dfg_2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2 = data.dfg_1, data.dfg_2, data.dfg_edge_index_1, data.dfg_edge_index_2, data.dfg_edge_attr_1, data.dfg_edge_attr_2
            label = data.clone_label
            label = transfer_label(label)
            label = torch.tensor(label, dtype=torch.float, device=device)

            ast1 = torch.tensor(ast_1, dtype=torch.long, device=device)
            ast2 = torch.tensor(ast_2, dtype=torch.long, device=device)
            ast_edge_index_1 = torch.tensor(ast_edge_index_1, dtype=torch.long, device=device)
            ast_edge_index_2 = torch.tensor(ast_edge_index_2, dtype=torch.long, device=device)
            ast_edge_attr_1 = torch.tensor(ast_edge_attr_1, dtype=torch.long, device=device)
            ast_edge_attr_2 = torch.tensor(ast_edge_attr_2, dtype=torch.long, device=device)

            cfg1 = torch.tensor(cfg_1, dtype=torch.long, device=device)
            cfg2 = torch.tensor(cfg_2, dtype=torch.long, device=device)
            cfg_edge_index_1 = torch.tensor(cfg_edge_index_1, dtype=torch.long, device=device)
            cfg_edge_index_2 = torch.tensor(cfg_edge_index_2, dtype=torch.long, device=device)
            cfg_edge_attr_1 = torch.tensor(cfg_edge_attr_1, dtype=torch.long, device=device)
            cfg_edge_attr_2 = torch.tensor(cfg_edge_attr_2, dtype=torch.long, device=device)
            
            dfg1 = torch.tensor(dfg_1, dtype=torch.long, device=device)
            dfg2 = torch.tensor(dfg_2, dtype=torch.long, device=device)
            dfg_edge_index_1 = torch.tensor(dfg_edge_index_1, dtype=torch.long, device=device)
            dfg_edge_index_2 = torch.tensor(dfg_edge_index_2, dtype=torch.long, device=device)
            dfg_edge_attr_1 = torch.tensor(dfg_edge_attr_1, dtype=torch.long, device=device)
            dfg_edge_attr_2 = torch.tensor(dfg_edge_attr_2, dtype=torch.long, device=device)

            data_ast = [ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2]
            data_cfg = [cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2]
            data_dfg = [dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2]
            data_input = [data_ast, data_cfg, data_dfg]

            prediction = model(data_input)
            output = F.cosine_similarity(prediction[0], prediction[1])
            results.append(output.item())
            prediction = torch.sign(output).item()

            if prediction > params['threshold']:
                prediction = int(1)
            else:
                prediction = int(0)

            bcb_samples.append((data, prediction))

    end_time = time.time()
    total = end_time - start_time
    log.info(f"Total evaluation time: {total:.2f} sec")
    bcb_evaluation(bcb_samples)


def train_mugret(dataset, params, net_params, dirs):
    root_log_dir, root_ckpt_dir = dirs

    def create_batches(data):
        batches = [data[graph:graph + params['batch_size']] for graph in range(0, len(data), params['batch_size'])]
        return batches

    vocablen, trainset, valset, testset, newtestset = dataset.vocab_length, dataset.train_data, dataset.val_data, dataset.test_data, dataset.new_test_data
    device = net_params['device']
    net_params['vocablen'] = vocablen
    log.info(f"Vocab length: {vocablen}")
    log.info(f"TrainSet length: {len(trainset)}")
    log.info(f"ValSet length: {len(valset)}")
    log.info(f"TestSet length: {len(testset)}")
    log.info(f"NewTestSet length: {len(newtestset)}")

    # Model setting
    log.info("Model setting")
    model = MuGReTNet(net_params)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    
    # Loss functions
    criterion = nn.MSELoss()

    epochs = trange(params['epochs'], leave=True, desc="Epoch")
    for epoch in epochs:
        model.train()  # Set the model to training mode
        batches = create_batches(trainset)
        total_loss = 0.0
        main_index = 0.0
        epoch_start = time.time()    
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            optimizer.zero_grad()
            batchloss = 0
            for data in batch:
                ast_1, ast_2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2 = data.ast_1, data.ast_2, data.ast_edge_index_1, data.ast_edge_index_2, data.ast_edge_attr_1, data.ast_edge_attr_2
                cfg_1, cfg_2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2 = data.cfg_1, data.cfg_2, data.cfg_edge_index_1, data.cfg_edge_index_2, data.cfg_edge_attr_1, data.cfg_edge_attr_2
                dfg_1, dfg_2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2 = data.dfg_1, data.dfg_2, data.dfg_edge_index_1, data.dfg_edge_index_2, data.dfg_edge_attr_1, data.dfg_edge_attr_2

                label = data.clone_label
                label = transfer_label(label)
                label = torch.tensor(label, dtype=torch.float, device=device)

                ast1 = torch.tensor(ast_1, dtype=torch.long, device=device)
                ast2 = torch.tensor(ast_2, dtype=torch.long, device=device)
                ast_edge_index_1 = torch.tensor(ast_edge_index_1, dtype=torch.long, device=device)
                ast_edge_index_2 = torch.tensor(ast_edge_index_2, dtype=torch.long, device=device)
                ast_edge_attr_1 = torch.tensor(ast_edge_attr_1, dtype=torch.long, device=device)
                ast_edge_attr_2 = torch.tensor(ast_edge_attr_2, dtype=torch.long, device=device)

                cfg1 = torch.tensor(cfg_1, dtype=torch.long, device=device)
                cfg2 = torch.tensor(cfg_2, dtype=torch.long, device=device)
                cfg_edge_index_1 = torch.tensor(cfg_edge_index_1, dtype=torch.long, device=device)
                cfg_edge_index_2 = torch.tensor(cfg_edge_index_2, dtype=torch.long, device=device)
                cfg_edge_attr_1 = torch.tensor(cfg_edge_attr_1, dtype=torch.long, device=device)
                cfg_edge_attr_2 = torch.tensor(cfg_edge_attr_2, dtype=torch.long, device=device)
                
                dfg1 = torch.tensor(dfg_1, dtype=torch.long, device=device)
                dfg2 = torch.tensor(dfg_2, dtype=torch.long, device=device)
                dfg_edge_index_1 = torch.tensor(dfg_edge_index_1, dtype=torch.long, device=device)
                dfg_edge_index_2 = torch.tensor(dfg_edge_index_2, dtype=torch.long, device=device)
                dfg_edge_attr_1 = torch.tensor(dfg_edge_attr_1, dtype=torch.long, device=device)
                dfg_edge_attr_2 = torch.tensor(dfg_edge_attr_2, dtype=torch.long, device=device)

                data_ast = [ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2]
                data_cfg = [cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2]
                data_dfg = [dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2]
                data = [data_ast, data_cfg, data_dfg]

                prediction = model(data)
                cossim = F.cosine_similarity(prediction[0], prediction[1])
                batchloss = batchloss + criterion(cossim, label)
            
            batchloss.backward(retain_graph=True)
            optimizer.step()
            loss = batchloss.item()
            total_loss += loss
            main_index += len(batch)
            loss = total_loss / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

        epoch_end = time.time()
        log.info(f"Epoch {epoch} finished in {epoch_end - epoch_start:.2f} sec")

        # Periodic evaluation and saving
        if epoch % params['eval_epoch_interval'] == 0:
            log.info(f"Start evaluation on testset in epoch: {epoch}")
            evaluation_mugret(model, testset, params, net_params)
        if epoch % params['save_epoch_interval'] == 0:
            log.info(f"Saving model in epoch: {epoch}")
            torch.save(model.state_dict(), f"{root_ckpt_dir}/model_{epoch}.pth")
