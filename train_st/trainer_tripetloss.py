from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset


# Load the full dataset
dataset = load_dataset("csv", data_files="triplet_dataset_new.csv")
print(type(dataset))
# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# Define loss
# loss = losses.TripletLoss(model=model)
loss = losses.MultipleNegativesRankingLoss(model=model,scale=60)
# Define training arguments
# training_args = SentenceTransformerTrainingArguments(
#     output_dir="/scratch/zczlyf7/st_models/MultipleNegativesRankingLoss",
#     overwrite_output_dir=True,
#     per_device_train_batch_size=128,
#     num_train_epochs=4,
#     warmup_steps=100,
#     save_strategy="epoch",         # Save every epoch
#     logging_steps=20,
#     save_total_limit=2,
#     remove_unused_columns=False,
#     fp16=True,
#     dataloader_num_workers=4,
#     do_train=True,
#     do_eval=True
# )


# Use best hyperparameters from Optuna
training_args = SentenceTransformerTrainingArguments(
    output_dir="/scratch/zczlyf7/st_models/MultipleNegativesRankingLoss/hpo_scale_60",
    overwrite_output_dir=True,
    per_device_train_batch_size=32,                     #  From best_trial
    num_train_epochs=2,                                 #  From best_trial
    warmup_ratio=0.03254893834779507,                   #  From best_trial
    learning_rate=2.1456771788455288e-05,               #  From best_trial
    save_strategy="epoch",                             
    logging_steps=20,
    save_total_limit=2,
    remove_unused_columns=False,                        # Required for triplet format
    fp16=True,
    dataloader_num_workers=4,
    do_train=True,
    do_eval=False,                                    
)


# Initialize trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss
)

# Start training
trainer.train()
