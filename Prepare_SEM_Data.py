# from super_image.data import DatasetBuilder
# DatasetBuilder.prepare(base_path = r'/media/tyler/748D862954B8472E/SEM_Dataset/Nanowires',output_path=r'/media/tyler/748D862954B8472E/SEM_Dataset/Built.h5')

from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop,TrainAugmentDatasetH5

# augmented_dataset = load_dataset(r'/media/tyler/748D862954B8472E/SEM_Dataset/Built.h5')\
#     .map(augment_five_crop, batched=True, desc="Augmenting Dataset")                                # download and augment the data with the five_crop method
train_dataset = TrainDataset(augmented_dataset)                                                     # prepare the train dataset for loading PyTorch DataLoader
# train_dataset = TrainAugmentDatasetH5(h5_file=r'/media/tyler/748D862954B8472E/SEM_Dataset/Built.h5',scale=4)
eval_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation'))      # prepare the eval dataset for the PyTorch DataLoader

from super_image import Trainer, TrainingArguments, DrlnModel, DrlnConfig

training_args = TrainingArguments(
    output_dir='./results',                 # output directory
    num_train_epochs=2,                  # total number of training epochs
    per_device_train_batch_size = 1
)

config = DrlnConfig(
    scale=4,                                # train a model to upscale 4x
)
model = DrlnModel(config)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

trainer.train()