import torch

from .dataset import DrugCellBatch


def drugcell_collate_fn(batch, labeled: bool):
    cell_ids, drug_ids = list(), list()
    if labeled:
        labels = list()
        for data in batch:
            cell_id, drug_id, label = data.cell_id, data.drug_id, data.label
            cell_ids.append(cell_id)
            drug_ids.append(drug_id)
            labels.append(label)
        cell_ids, drug_ids, labels = torch.Tensor(cell_ids).int(
        ), torch.Tensor(drug_ids).int(), torch.Tensor(labels)
        return DrugCellBatch(cell_ids, drug_ids, labels.unsqueeze_(-1))
    else:
        for data in batch:
            cell_id, drug_id = data.cell_id, data.drug_id
            cell_ids.append(cell_id)
            drug_ids.append(drug_id)
        cell_ids, drug_ids = torch.Tensor(
            cell_ids).int(), torch.Tensor(drug_ids).int()
        return DrugCellBatch(cell_ids, drug_ids)
