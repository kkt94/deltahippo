import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from utils.wrapmodel import WrapModel
from utils.backbone import obtain_features

logger = logging.getLogger()

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=500, latent_dim=10):
        super(SimpleAutoencoder, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

def target_distribution(q):
    # Compute the target distribution p
    weight = (q ** 2) / q.sum(0)
    return (weight.T / weight.sum(1)).T

def evaluate_deep_clustering_on_all_task(params, task_id, CL_dataset, test_loader_list, model, tokenizer, accelerator) -> dict:
    """
    Performs clustering using a Deep Clustering algorithm (e.g., DEC) and evaluates the performance.
    """
    il_mode = params.il_mode
    assert il_mode in ['CIL', 'TIL'], 'NotImplemented for il_mode %s' % (il_mode)

    total_num_task = CL_dataset.continual_config['NUM_TASK']
    cur_num_class_list = CL_dataset.continual_config['CUR_NUM_CLASS']

    if il_mode == 'CIL':
        num_task = len(test_loader_list)
        cur_num_class = cur_num_class_list
        num_class = sum(cur_num_class)
    else:
        num_task = len(test_loader_list)
        cur_num_class = cur_num_class_list[:num_task]
        num_class = cur_num_class

    # Prepare the model
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, WrapModel):
        model = model.model
    if hasattr(model, 'backbone_model'):
        model = model.backbone_model

    # Step 1: Feature extraction
    with torch.no_grad():
        test_feature_list = []
        test_label_idx_list = []
        for t_id in range(num_task):
            _test_feature_list = []
            _test_label_idx_list = []
            for lm_input in test_loader_list[t_id]:
                extracted_features = obtain_features(params=params,
                                                     model=model,
                                                     lm_input=lm_input,
                                                     tokenizer=tokenizer)
                if il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']

                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                extracted_features = extracted_features.detach().cpu()
                label_idx = label_idx.cpu()

                # Handle word-level tasks
                if extracted_features.dim() == 3:
                    batch_size, seq_length, feature_dim = extracted_features.shape
                    extracted_features = extracted_features.view(-1, feature_dim)
                    label_idx = label_idx.view(-1)
                    mask = label_idx != -100  # Exclude padding tokens
                    extracted_features = extracted_features[mask]
                    label_idx = label_idx[mask]

                _test_feature_list.append(extracted_features)
                _test_label_idx_list.append(label_idx)

            _test_feature_list = torch.cat(_test_feature_list, dim=0)
            _test_label_idx_list = torch.cat(_test_label_idx_list, dim=0)

            test_feature_list.append(_test_feature_list)
            test_label_idx_list.append(_test_label_idx_list)

        if il_mode == 'CIL':
            test_features_all = torch.cat(test_feature_list, dim=0)
            test_label_idx_all = torch.cat(test_label_idx_list, dim=0)
        else:
            test_features_all = test_feature_list
            test_label_idx_all = test_label_idx_list

        del test_feature_list, test_label_idx_list
        torch.cuda.empty_cache()

    # Deep clustering and evaluation
    if accelerator.is_main_process:
        deep_clustering_result_dict = {}
        if il_mode == 'CIL':
            features = test_features_all
            labels = test_label_idx_all.numpy()
            num_clusters = sum(cur_num_class)
            print(f"CIL_num_data: {len(labels)}")
            print(f"CIL_num_clusters: {num_clusters}")

            # Initialize the autoencoder
            input_dim = features.shape[1]
            hidden_dim = 500
            latent_dim = num_clusters  # Set latent_dim to the number of clusters
            autoencoder = SimpleAutoencoder(input_dim, hidden_dim, latent_dim).to(accelerator.device)

            # Pretrain the autoencoder
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(features)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

            pretrain_epochs = 10
            for epoch in range(pretrain_epochs):
                autoencoder.train()
                total_loss = 0
                for batch_features in dataloader:
                    batch_features = batch_features[0].to(accelerator.device)
                    _, recon = autoencoder(batch_features)
                    loss = criterion(recon, batch_features)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_features.size(0)
                print(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}], Loss: {total_loss / len(dataset):.4f}")

            # Initialize cluster centers using K-Means
            autoencoder.eval()
            with torch.no_grad():
                latent_features, _ = autoencoder(features.to(accelerator.device))
                latent_features = latent_features.cpu().numpy()

            kmeans = KMeans(n_clusters=num_clusters, n_init=20)
            y_pred = kmeans.fit_predict(latent_features)
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(accelerator.device)

            # Train with the clustering loss
            max_iter = 100
            update_interval = 10
            tol = 1e-3

            # Soft assignment computation function
            def compute_q(z, cluster_centers):
                q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2))
                q = q ** ((1.0 + 1.0) / 2.0)
                q = (q.t() / torch.sum(q, dim=1)).t()
                return q

            # Prepare the data loader
            dataset = torch.utils.data.TensorDataset(features)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

            index_array = np.arange(features.shape[0])
            for ite in range(max_iter):
                # Update the target distribution at regular intervals
                if ite % update_interval == 0:
                    autoencoder.eval()
                    latent_features_all = []
                    with torch.no_grad():
                        for batch_features in dataloader:
                            batch_features = batch_features[0].to(accelerator.device)
                            z, _ = autoencoder(batch_features)
                            latent_features_all.append(z)
                    latent_features = torch.cat(latent_features_all, dim=0)
                    q = compute_q(latent_features, cluster_centers)
                    p = target_distribution(q.detach().cpu().numpy())
                    p = torch.tensor(p, dtype=torch.float32).to(accelerator.device)

                    # Compute the KL divergence loss
                    kl_loss = nn.KLDivLoss(reduction='batchmean')
                    y_pred_last = y_pred
                    y_pred = torch.argmax(q, dim=1).cpu().numpy()
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    print(f"Iteration {ite}, delta_label: {delta_label:.4f}")
                    if delta_label < tol:
                        print("Converged")
                        break

                # Per-batch training
                autoencoder.train()
                total_loss = 0
                for batch_idx, (batch_features,) in enumerate(dataloader):
                    batch_features = batch_features.to(accelerator.device)
                    batch_size_i = batch_features.size(0)
                    idx = index_array[batch_idx * batch_size_i: (batch_idx + 1) * batch_size_i]
                    z, x_recon = autoencoder(batch_features)
                    q_batch = compute_q(z, cluster_centers)
                    p_batch = p[idx]
                    kl = kl_loss(torch.log(q_batch), p_batch)
                    recon_loss = criterion(x_recon, batch_features)
                    loss = kl + recon_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_size_i
                print(f"Iteration {ite}, Loss: {total_loss / len(dataset):.4f}")

            # Final clustering
            autoencoder.eval()
            with torch.no_grad():
                latent_features_all = []
                for batch_features in dataloader:
                    batch_features = batch_features[0].to(accelerator.device)
                    z, _ = autoencoder(batch_features)
                    latent_features_all.append(z)
                latent_features = torch.cat(latent_features_all, dim=0)
                q = compute_q(latent_features, cluster_centers)
                y_pred = torch.argmax(q, dim=1).cpu().numpy()

            # Map cluster labels to ground-truth labels
            le_labels = LabelEncoder()
            le_clusters = LabelEncoder()
            labels_encoded = le_labels.fit_transform(labels)
            clusters_encoded = le_clusters.fit_transform(y_pred)

            contingency_matrix = metrics.cluster.contingency_matrix(labels_encoded, clusters_encoded)
            row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

            label_mapping = {}
            for row, col in zip(row_ind, col_ind):
                cluster_label = le_clusters.inverse_transform([col])[0]
                true_label = le_labels.inverse_transform([row])[0]
                label_mapping[cluster_label] = true_label

            cluster_labels = np.unique(y_pred)
            missing_labels = set(cluster_labels) - set(label_mapping.keys())
            for missing_label in missing_labels:
                label_mapping[missing_label] = -1

            mapped_clusters = np.array([label_mapping.get(c, -1) for c in y_pred])

            valid_indices = mapped_clusters != -1
            accuracy = np.mean(mapped_clusters[valid_indices] == labels[valid_indices]) * 100
            accuracy = np.round(accuracy, 3)
            ari = metrics.adjusted_rand_score(labels[valid_indices], mapped_clusters[valid_indices]) * 100
            ari = np.round(ari, 4)
            nmi = metrics.normalized_mutual_info_score(labels[valid_indices], mapped_clusters[valid_indices]) * 100
            nmi = np.round(nmi, 4)

            deep_clustering_result_dict['Overall_Accuracy'] = accuracy
            deep_clustering_result_dict['Overall_ARI'] = ari
            deep_clustering_result_dict['Overall_NMI'] = nmi
            print(deep_clustering_result_dict)

            # Per-task metrics are not computed in CIL mode

        else:
            # TIL mode is not implemented
            pass

    else:
        # If not the main process, fill the results with -1
        num_tasks_for_results = total_num_task
        deep_clustering_result_dict = {}
        for t_id in range(num_tasks_for_results):
            deep_clustering_result_dict[f'Task_{t_id}_Accuracy'] = -1
            deep_clustering_result_dict[f'Task_{t_id}_ARI'] = -1
            deep_clustering_result_dict[f'Task_{t_id}_NMI'] = -1
        if il_mode == 'CIL':
            deep_clustering_result_dict['Overall_Accuracy'] = -1
            deep_clustering_result_dict['Overall_ARI'] = -1
            deep_clustering_result_dict['Overall_NMI'] = -1

    return deep_clustering_result_dict

