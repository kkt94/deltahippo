import logging
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from seqeval.metrics import f1_score
from random import shuffle
from copy import deepcopy

from utils.wrapmodel import WrapModel
from utils.backbone import obtain_features
from utils.classifier import CosineLinear

logger = logging.getLogger()

def probing_on_all_task(params, task_id, CL_dataset, train_loader_list, test_loader_list, model, tokenizer, accelerator) -> dict:
    """
        Probing: It re-trains classifiers on data of ALL tasks (CIL mode) or up to the next task (TIL mode).
        The probing performance represents what PTMs really know about the IL tasks.

        Return:
            - prob_result_dict: {
                'LinearProb': [list of per-task accuracies],  # Linear Probing performance
                # Results of other classifiers are included as well
            }
    """
    il_mode = params.il_mode
    assert il_mode in ['CIL', 'TIL'], 'NotImplemented for il_mode %s' % (il_mode)

    total_num_task = CL_dataset.continual_config['NUM_TASK']
    cur_num_class_list = CL_dataset.continual_config['CUR_NUM_CLASS']

    if il_mode == 'CIL':
        # In CIL mode, use data from all tasks
        num_task = len(train_loader_list)  # Actual length of train_loader_list (all tasks)
        cur_num_class = cur_num_class_list  # List of class counts for all tasks
        num_class = sum(cur_num_class)
    else:
        # In TIL mode, use data up to the current and next task
        num_task = len(train_loader_list)  # Length of the given train_loader_list
        cur_num_class = cur_num_class_list[:num_task]
        num_class = cur_num_class  # List of per-task class counts

    # Prepare the model
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, WrapModel):
        model = model.model
    if hasattr(model, 'backbone_model'):
        model = model.backbone_model

    # Step 1: Extract all features
    with torch.no_grad():
        train_feature_list = []
        train_label_idx_list = []
        for t_id in range(num_task):
            _train_feature_list = []
            _train_label_idx_list = []
            for lm_input in train_loader_list[t_id]:
                extracted_features = obtain_features(params=params,
                                                     model=model,
                                                     lm_input=lm_input,
                                                     tokenizer=tokenizer)
                if il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']

                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                _train_feature_list.append(extracted_features.detach().cpu())
                _train_label_idx_list.append(label_idx.cpu())

            _train_feature_list = torch.cat(_train_feature_list, dim=0)
            _train_label_idx_list = torch.cat(_train_label_idx_list, dim=0)

            train_feature_list.append(_train_feature_list)
            train_label_idx_list.append(_train_label_idx_list)

        if il_mode == 'CIL':
            train_features_all = torch.cat(train_feature_list, dim=0)
            train_label_idx_all = torch.cat(train_label_idx_list, dim=0)
        else:
            train_features_all = train_feature_list
            train_label_idx_all = train_label_idx_list

        del train_feature_list, train_label_idx_list
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        # Step 2: Train the classifiers
        if il_mode == 'CIL':
            train_features = train_features_all
            train_label_idx = train_label_idx_all

            feature_dim = train_features.shape[-1]

            # Initialize classifiers (output dim is the total number of classes)
            linear_layer = nn.Linear(in_features=feature_dim, out_features=num_class, bias=False).cuda().train()
            coslinear_layer = CosineLinear(in_features=feature_dim, out_features=num_class).cuda().train()

            prototype_layer = nn.Linear(in_features=feature_dim, out_features=num_class, bias=False).cuda()
            cosprototype_layer = CosineLinear(in_features=feature_dim, out_features=num_class).cuda()

            opt_linear = Adam(linear_layer.parameters(), lr=0.001)
            opt_coslinear = Adam(coslinear_layer.parameters(), lr=0.001)
            loss_fct = nn.CrossEntropyLoss()
            num_chunk = max(train_features.shape[0] // 128, 1)
            loss_list_epochs_linear = []
            loss_list_epochs_coslinear = []

            # Initialize variables for prototype computation
            cur_class_prototypes = torch.zeros_like(prototype_layer.weight.data).cuda()
            cnt_class_samples = {class_idx: 0 for class_idx in range(num_class)}

            for e_id in range(20):

                loss_list_linear = []
                loss_list_coslinear = []
                shuffle_idx = torch.randperm(train_features.shape[0])

                for selected_idx in torch.chunk(shuffle_idx, num_chunk):
                    _feature, _label_idx = train_features[selected_idx], train_label_idx[selected_idx]
                    _feature, _label_idx = _feature.cuda(), _label_idx.cuda()

                    if params.classification_type == 'word-level':
                        label_mask = (_label_idx != -100)
                        _feature = _feature[label_mask]  # flatten
                        _label_idx = _label_idx[label_mask]  # flatten

                    # Train Linear Classifier
                    logits_linear = linear_layer(_feature)
                    loss_linear = loss_fct(logits_linear, _label_idx)
                    opt_linear.zero_grad()
                    loss_linear.backward()
                    opt_linear.step()

                    # Train Cosine Linear Classifier
                    logits_coslinear = coslinear_layer(_feature)
                    loss_coslinear = loss_fct(logits_coslinear, _label_idx)
                    opt_coslinear.zero_grad()
                    loss_coslinear.backward()
                    opt_coslinear.step()

                    loss_list_linear.append(loss_linear.item())
                    loss_list_coslinear.append(loss_coslinear.item())

                    # Compute Class Center for Prototype/CosinePrototype Classifier (Only need one epoch)
                    if e_id == 0:
                        for class_idx in torch.unique(_label_idx):
                            class_mask = (_label_idx == class_idx)
                            cnt_class_samples[class_idx.item()] += class_mask.sum().item()
                            cur_class_prototypes[class_idx.item()] += _feature[class_mask].detach().sum(dim=0)

                loss_list_epochs_linear.append(np.mean(loss_list_linear))
                loss_list_epochs_coslinear.append(np.mean(loss_list_coslinear))

            # Set the weight of the Prototype/CosinePrototype Classifier
            for class_idx in cnt_class_samples.keys():
                if cnt_class_samples[class_idx] == 0:
                    continue
                prototype_layer.weight.data[class_idx] = cur_class_prototypes[class_idx] / cnt_class_samples[class_idx]
                cosprototype_layer.weight.data[class_idx] = cur_class_prototypes[class_idx] / cnt_class_samples[class_idx]

            del train_features, train_label_idx
            torch.cuda.empty_cache()

        else:
            # TIL mode
            feature_dim = train_features_all[0].shape[-1]

            linear_layer = nn.ModuleList([
                nn.Linear(in_features=feature_dim, out_features=cur_num_class[t_id], bias=False).cuda().train()
                for t_id in range(num_task)
            ])
            coslinear_layer = nn.ModuleList([
                CosineLinear(in_features=feature_dim, out_features=cur_num_class[t_id]).cuda().train()
                for t_id in range(num_task)
            ])

            prototype_layer = nn.ModuleList([
                nn.Linear(in_features=feature_dim, out_features=cur_num_class[t_id], bias=False).cuda()
                for t_id in range(num_task)
            ])
            cosprototype_layer = nn.ModuleList([
                CosineLinear(in_features=feature_dim, out_features=cur_num_class[t_id]).cuda()
                for t_id in range(num_task)
            ])

            opt_linear_list = [Adam(clf.parameters(), lr=0.001) for clf in linear_layer]
            opt_coslinear_list = [Adam(clf.parameters(), lr=0.001) for clf in coslinear_layer]
            loss_fct = nn.CrossEntropyLoss()

            # Initialize variables for prototype computation
            cur_class_prototypes = [
                torch.zeros_like(prototype_layer[t_id].weight.data).cuda()
                for t_id in range(num_task)
            ]
            cnt_class_samples = [
                {class_idx: 0 for class_idx in range(cur_num_class[t_id])}
                for t_id in range(num_task)
            ]

            for t_id in range(num_task):
                train_features = train_features_all[t_id]
                train_label_idx = train_label_idx_all[t_id]

                num_chunk = max(train_features.shape[0] // 128, 1)
                loss_list_epochs_linear = []
                loss_list_epochs_coslinear = []

                for e_id in range(20):

                    loss_list_linear = []
                    loss_list_coslinear = []
                    shuffle_idx = torch.randperm(train_features.shape[0])

                    for selected_idx in torch.chunk(shuffle_idx, num_chunk):
                        _feature, _label_idx = train_features[selected_idx], train_label_idx[selected_idx]
                        _feature, _label_idx = _feature.cuda(), _label_idx.cuda()

                        if params.classification_type == 'word-level':
                            label_mask = (_label_idx != -100)
                            _feature = _feature[label_mask]  # flatten
                            _label_idx = _label_idx[label_mask]  # flatten

                        # Train Linear Classifier
                        logits_linear = linear_layer[t_id](_feature)
                        loss_linear = loss_fct(logits_linear, _label_idx)
                        opt_linear_list[t_id].zero_grad()
                        loss_linear.backward()
                        opt_linear_list[t_id].step()

                        # Train Cosine Linear Classifier
                        logits_coslinear = coslinear_layer[t_id](_feature)
                        loss_coslinear = loss_fct(logits_coslinear, _label_idx)
                        opt_coslinear_list[t_id].zero_grad()
                        loss_coslinear.backward()
                        opt_coslinear_list[t_id].step()

                        loss_list_linear.append(loss_linear.item())
                        loss_list_coslinear.append(loss_coslinear.item())

                        # Compute Class Center for Prototype/CosinePrototype Classifier (Only need one epoch)
                        if e_id == 0:
                            for class_idx in torch.unique(_label_idx):
                                class_mask = (_label_idx == class_idx)
                                cnt_class_samples[t_id][class_idx.item()] += class_mask.sum().item()
                                cur_class_prototypes[t_id][class_idx.item()] += _feature[class_mask].detach().sum(dim=0)

                    loss_list_epochs_linear.append(np.mean(loss_list_linear))
                    loss_list_epochs_coslinear.append(np.mean(loss_list_coslinear))

                # Set the weight of the Prototype/CosinePrototype Classifier
                for class_idx in cnt_class_samples[t_id].keys():
                    if cnt_class_samples[t_id][class_idx] == 0:
                        continue
                    prototype_layer[t_id].weight.data[class_idx] = cur_class_prototypes[t_id][class_idx] / cnt_class_samples[t_id][class_idx]
                    cosprototype_layer[t_id].weight.data[class_idx] = cur_class_prototypes[t_id][class_idx] / cnt_class_samples[t_id][class_idx]

                del train_features, train_label_idx
                torch.cuda.empty_cache()

            del train_features_all, train_label_idx_all

        # Step 3: Evaluate the classifiers
        with torch.no_grad():
            prob_result_dict = {}
            for metric_name, _classifier in zip(['LinearProb', 'CosineLinearProb', 'PrototypeProb', 'CosinePrototypeProb'],
                                                [linear_layer, coslinear_layer, prototype_layer, cosprototype_layer]):
                if il_mode == 'CIL':
                    _classifier.eval()
                else:
                    for clf in _classifier:
                        clf.eval()

                if params.classification_type == 'sentence-level':
                    tasks_acc_list = []
                    for t_id in range(total_num_task):
                        if il_mode == 'TIL' and t_id >= num_task:
                            tasks_acc_list.append(-1)  # Mark tasks not included in TIL mode as -1
                            continue
                        acc_list = []
                        for lm_input in test_loader_list[t_id]:
                            _feature = obtain_features(params=params,
                                                       model=model,
                                                       lm_input=lm_input,
                                                       tokenizer=tokenizer)
                            if il_mode == 'CIL':
                                _label_idx = lm_input['label_idx_cil']
                                _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
                                logits = _classifier(_feature)
                            else:
                                _label_idx = lm_input['label_idx_til']
                                _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
                                logits = _classifier[t_id](_feature)
                            preds = logits.argmax(dim=-1).detach().cpu()
                            _label_idx = _label_idx.cpu()
                            acc_list.extend([(_pred == _label).item() for _pred, _label in zip(preds, _label_idx)])
                        if len(acc_list) > 0:
                            task_accuracy = np.round(np.mean(acc_list) * 100, 3)
                        else:
                            task_accuracy = -1  # Mark as -1 when no data is available
                        tasks_acc_list.append(task_accuracy)
                        torch.cuda.empty_cache()
                else:
                    # Word-level probing is not implemented
                    raise NotImplementedError('Word-level classification not implemented for this scenario')

                prob_result_dict[metric_name] = deepcopy(tasks_acc_list)

        # Save the classifiers if needed
        if params.save_probing_classifiers:
            if il_mode == 'CIL':
                torch.save(
                    {
                        'linear_layer': linear_layer.cpu(),
                        'coslinear_layer': coslinear_layer.cpu(),
                        'prototype_layer': prototype_layer.cpu(),
                        'cosprototype_layer': cosprototype_layer.cpu(),
                    },
                    os.path.join(params.dump_path, 'probing_classifiers_task%d.pth' % (task_id))
                )
            else:
                torch.save(
                    {
                        'linear_layer': [layer.cpu() for layer in linear_layer],
                        'coslinear_layer': [layer.cpu() for layer in coslinear_layer],
                        'prototype_layer': [layer.cpu() for layer in prototype_layer],
                        'cosprototype_layer': [layer.cpu() for layer in cosprototype_layer],
                    },
                    os.path.join(params.dump_path, 'probing_classifiers_task%d.pth' % (task_id))
                )

    else:
        # Non-main processes return placeholder results
        num_tasks_for_results = total_num_task
        prob_result_dict = {
            'LinearProb': [-1] * num_tasks_for_results,
            'CosineLinearProb': [-1] * num_tasks_for_results,
            'PrototypeProb': [-1] * num_tasks_for_results,
            'CosinePrototypeProb': [-1] * num_tasks_for_results,
        }

    return prob_result_dict


def save_all_features_labels(params, save_name, train_loader_list, model, tokenizer, accelerator) -> None:
    '''
        Save all features and labels
    '''
    il_mode = params.il_mode
    assert il_mode in ['CIL','TIL'], 'NotImplemented for il_mode %s'%(il_mode)
    
    num_task = len(train_loader_list)
    with torch.no_grad():
        train_feature_list = []
        train_label_idx_list = []
        for t_id in range(num_task):
            _train_feature_list = []
            _train_label_idx_list = []
            for lm_input in train_loader_list[t_id]:  
                extracted_features = obtain_features(params=params, 
                                                        model=model, 
                                                        lm_input=lm_input, 
                                                        tokenizer=tokenizer)
                if params.il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']
                
                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                _train_feature_list.append(extracted_features.detach().cpu())
                _train_label_idx_list.append(label_idx.cpu())

            _train_feature_list = torch.cat(_train_feature_list, dim=0)
            _train_label_idx_list = torch.cat(_train_label_idx_list, dim=0)

            train_feature_list.append(_train_feature_list)
            train_label_idx_list.append(_train_label_idx_list)

        if params.il_mode == 'CIL':
            train_features_all = torch.cat(train_feature_list, dim=0)
            train_label_idx_all = torch.cat(train_label_idx_list, dim=0)
        else:
            train_features_all = train_feature_list
            train_label_idx_all = train_label_idx_list

        save_path = os.path.join(params.dump_path,'train_features_labels_%s.pth'%(save_name))
        torch.save(
            {'features':train_features_all,'labels':train_label_idx_all},
            save_path
        )
        
        del train_feature_list, train_label_idx_list
        accelerator.free_memory()