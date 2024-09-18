from argparse import ArgumentParser
import yaml
import torch
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from gan_module import AgingGAN

parser = ArgumentParser()
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')

def main():
    args = parser.parse_args()

    try:
        with open(args.config) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing the configuration file: {e}")
        return

    print("Configuration loaded:", config)
    print(f"MPS Available: {torch.backends.mps.is_available()}, CUDA Available: {torch.cuda.is_available()}")

    model = AgingGAN(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='AgingGAN-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    logger = pl_loggers.TensorBoardLogger("training_logs", name="AgingGAN")

    if torch.backends.mps.is_available():
        accelerator = 'mps'
        devices = 1
    elif torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
    else:
        accelerator = 'cpu'
        devices = None

    trainer = Trainer(
    max_epochs=config['epochs'],
    accelerator='auto',  # Let PyTorch Lightning choose the best available (auto might pick CPU/GPU)
    devices=1 if torch.cuda.is_available() else None,  # Use None for CPU
    auto_scale_batch_size='binsearch',
    callbacks=[checkpoint_callback],
    logger=logger
)

    

    trainer.fit(model)

if __name__ == '__main__':
    main()

