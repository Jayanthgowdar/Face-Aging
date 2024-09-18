# Face-Aging
This File holds code for a face aging deep learning model. It is based on the CycleGAN, where we translate young faces to old and vice versa with limited computational resources .

# Demo
To try out the pretrained model on your images, use the following command:
```bash
python infer.py --image_dir /Users/jayanth/Face-Aging/Face-Aging/archive/UTKFace
```

# Training
To train UTK faces datasets, we use the provided preprocessing scripts in the preprocessing directory to prepare the dataset.

using UTK faces, use the following:
```bash
python preprocessing/preprocess_utk.py --data_dir /Users/jayanth/Face-Aging/Face-Aging/archive/UTKFace --output_dir /Users/jayanth/Face-Aging/Face-Aging/FaceProcessed
```

Once the dataset is processed, we accessed ``` configs/aging_gan.yaml``` and modify the paths to point to the processed dataset you just created. then we ran training with:
```bash
python main.py
```

# Tensorboard
While training is running, you can observe the losses and the gan generated images in tensorboard, just point it to the 'lightning_logs' directory like so:
```bash
tensorboard --logdir=lightning_logs --bind_all
```
