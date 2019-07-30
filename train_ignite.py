from preprocess import prepare_data_ds

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping

import torch.utils.data as D

from models.ignite_trainer import DensenetTrainer
from torchvision.models import densenet121


path_data = "../DATA/kaggle-2019Q3-cellular/"
device = "cuda"
batch_size = 8
lr = 0.01
gamma = 0.99
num_workers = 6
backbone = densenet121
model_name = "densenet121_ignite"
IGTrainer = DensenetTrainer(model=backbone, lr=lr, gamma=gamma)
checkpoint_file = path_data + "models/" + model_name + ".ckpt"

if __name__ == "__main__":
    ds, ds_val, ds_test, df_train, df_val, df_test = prepare_data_ds(path_data)
    loader = D.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = D.DataLoader(
        ds_val, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    tloader = D.DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    trainer = create_supervised_trainer(
        IGTrainer.model, IGTrainer.optimizer, IGTrainer.criterion, device=device
    )
    val_evaluator = create_supervised_evaluator(
        IGTrainer.model, metrics=IGTrainer.metrics, device=device
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_val_metrics(engine):
        metrics = val_evaluator.run(val_loader).metrics
        print(
            "Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} ".format(
                engine.state.epoch, metrics["loss"], metrics["accuracy"]
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr_scheduler(engine):
        IGTrainer.lr_scheduler.step()
        lr = float(IGTrainer.optimizer.param_groups[0]["lr"])
        print("Learning rate: {}".format(lr))

    def logging_board(model_name="densenet121"):
        from ignite.contrib.handlers.tensorboard_logger import (
            TensorboardLogger,
            OutputHandler,
            OptimizerParamsHandler,
            GradsHistHandler,
        )

        tb_logger = TensorboardLogger("board/" + model_name)
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(
                tag="training", output_transform=lambda loss: {"loss": loss}
            ),
            event_name=Events.ITERATION_COMPLETED,
        )

        tb_logger.attach(
            val_evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=["accuracy", "loss"],
                another_engine=trainer,
            ),
            event_name=Events.EPOCH_COMPLETED,
        )

        tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(IGTrainer.optimizer),
            event_name=Events.ITERATION_STARTED,
        )

        tb_logger.attach(
            trainer,
            log_handler=GradsHistHandler(IGTrainer.model),
            event_name=Events.EPOCH_COMPLETED,
        )
        tb_logger.close()

    handler = EarlyStopping(
        patience=5,
        score_function=lambda engine: engine.state.metrics["accuracy"],
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.COMPLETED, handler)
    pbar = ProgressBar(bar_format="")
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})
    logging_board(model_name)

    from tqdm import tqdm
    from multiprocessing import Lock

    tqdm.set_lock(Lock())
    trainer.run(loader, max_epochs=5)
    checkpoint_epochs_file = ".".join(checkpoint_file.split(".")[:-1]) + ".05.ckpt"
    trainer.save_model(checkpoint_epochs_file)
    trainer.run(loader, max_epochs=10)
    checkpoint_epochs_file = ".".join(checkpoint_file.split(".")[:-1]) + ".15.ckpt"
    trainer.save_model(checkpoint_epochs_file)
