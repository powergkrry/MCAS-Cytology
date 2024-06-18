from config import get_config
import os
conf, unparsed = get_config()
if conf.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_number)

from torchvision import transforms
from utils import setup_torch, get_covid_transforms, load_model
from dataloader import load_all_patients
from imagenet import get_model
from multi_instance import SimpleMIL
from mil_trainer import ClassificationTrainer
from torch import optim
import warnings




def main(config):
    """
    Train a multi-instance classifier to detect covid positive vs. negative
    :return:
    """
    if not os.path.exists(os.path.join('/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models', config.experiment_name)):
            os.makedirs(os.path.join('/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models', config.experiment_name))
    # enforce batch_size of 1
    if config.batch_size != 1:
        warnings.warn("Batch size must be one for multi-instance learning, changing batch_size to 1")
        config.batch_size = 1

    setup_torch(config.random_seed, config.use_gpu, config.gpu_number)
    data_transforms = get_covid_transforms()

    train_loader, test_loader = load_all_patients(train_transforms=data_transforms['train'],
                                                  test_transforms=data_transforms['test'],
                                                  batch_size=1,
                                                  mil_size=config.mil_size,
                                                  test_data_index=config.test_data_index,
                                                  test_data_fold=config.test_data_fold)
    model = SimpleMIL(
        backbone_name=config.model_name, pretrained_backbone=config.pretrained_model


    )

    if config.use_gpu:
        model.cuda()
   

    if config.lr_schedule == 'plateau':
        #optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9)
        optimizer = optim.AdamW(model.parameters(), lr=config.init_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, mode='max', factor=0.316,
                                                         verbose=True)
    elif config.lr_schedule == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular2',
                                                step_size_up=10)
        # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=3e-1, mode='triangular',
        #                                        step_size_up=2000)
    else:
        scheduler = None
        if config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config.init_lr)
        elif config.optimizer.lower() == 'madgrad':
            from madgrad import MADGRAD
            optimizer = MADGRAD(model.parameters(), lr=config.init_lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.init_lr)
 
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader,
                                    batch_size=config.batch_size, epochs=config.epochs, patience=15, scheduler=scheduler, schedule_type=config.lr_schedule,
                                    run_name=config.experiment_name, test_data_index=config.test_data_index)
    trainer.train()

if __name__ == "__main__":
    conf, unparsed = get_config()

    main(conf)