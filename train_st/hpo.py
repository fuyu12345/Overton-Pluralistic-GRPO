from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers
from datasets import load_dataset

# Load your CSV dataset with (anchor, positive)
full_dataset = load_dataset("csv", data_files="triplet_dataset_new.csv")["train"]
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# Define the hyperparameter search space
def hpo_search_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [32, 64, 128]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        
    }

#Model reinitialization per trial
def hpo_model_init(trial):
    return SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# Step 4: Loss function init
def hpo_loss_init(model):
   
    return losses.MultipleNegativesRankingLoss(model)

def hpo_compute_objective(metrics):
    return -metrics["eval_loss"]  # because Optuna expects you to maximize


# Training arguments 
training_args = SentenceTransformerTrainingArguments(
    output_dir="/scratch/zczlyf7/st_models/hpo",
    overwrite_output_dir=True,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",
    do_eval=True,  
    save_strategy="no",
    logging_steps=10,
    run_name="mnr-csv-hpo"
)

# Create the trainer
trainer = SentenceTransformerTrainer(
    model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    model_init=hpo_model_init,
    loss=hpo_loss_init,
)

# Run hyperparameter search
best_trial = trainer.hyperparameter_search(
    hp_space=hpo_search_space,
    compute_objective=hpo_compute_objective,
    n_trials=20,
    direction="maximize",
    backend="optuna",
)

print("Best trial found:")
print(best_trial)
