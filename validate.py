import sys
import importlib
from train import ModelLightning
import models
import torch
import datasets
import pytorch_lightning as pyl
import numpy as np
import os
import sklearn.metrics


if __name__ == '__main__':
    config_file = sys.argv[1]
    ckpt_file = sys.argv[2]
    config = importlib.import_module(config_file).config

    backbone = models.HierarchicalTransformer(config)
    model = ModelLightning(
        config, backbone=backbone)

    model.load_state_dict(torch.load(ckpt_file)['state_dict'])
    model.eval()

    dataset = datasets.DygDatasetTest(config, 'val')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=datasets.dyg_test_collate_fn
        )

    trainer = pyl.Trainer(
        gpus=[2]
    )

    with torch.no_grad():
        pred = trainer.predict(
            model, dataloader)
        pass

    pred = np.hstack(pred)
    label = np.load(
        os.path.join(config['dataset_path'], 'val_labels.npy'))

    print(sklearn.metrics.roc_auc_score(label, pred))

    val_index = np.load(
        os.path.join(config['dataset_path'], 'val_index.npy'))

    pred[val_index[:, 1]==-1] = 0
    # pred.fill(0)

    print(sklearn.metrics.roc_auc_score(label, pred))
    pass
