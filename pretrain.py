# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from clearml import Task, Dataset

task = Task.init(project_name='nemo_sslr', task_name='conformer_large_pretrain_id_ms', output_uri='s3://experiment-logging/storage', task_type='training')
task.add_tags(['NeMo Toolkit', 'Pretraining', 'ASR', 'Voxlingua'])
task.set_base_docker('dleongsh/nemo_asr:v1.6.2')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

import os
from utils import update_manifest_from_json
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
# Example of unsupervised pre-training of a model
```sh
python speech_pre_training.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    trainer.gpus=-1 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Namex of project>"
```
For documentation on fine-tuning, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations
When doing supervised fine-tuning from unsupervised pre-trained encoder, set flag init_strict to False
"""


@hydra_runner(config_path="./configs", config_name="id_ms_sslr")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # download data
    train_manifest_paths = []
    validation_manifest_paths = []
    for dataset_name in cfg.s3.dataset_names:
        dataset = Dataset.get(dataset_project=cfg.s3.dataset_project, dataset_name=dataset_name)
        dataset_path = dataset.get_local_copy()
        train_manifest_paths.append(
            # updates the audio_filepaths in the manifest to the remote's full path to each audio file
            # outputs the updated path to train_manifest.json 
            update_manifest_from_json(os.path.join(dataset_path, 'train_manifest.json'))
            )
        validation_manifest_paths.append(
            # updates the audio_filepaths in the manifest to the remote's full path to each audio file
            # outputs the updated path to dev_manifest.json 
            update_manifest_from_json(os.path.join(dataset_path, 'dev_manifest.json'))
            )

    cfg.model.train_ds.manifest_filepath = ','.join(train_manifest_paths)
    cfg.model.validation_ds.manifest_filepath = ','.join(validation_manifest_paths)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = SpeechEncDecSelfSupervisedModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()