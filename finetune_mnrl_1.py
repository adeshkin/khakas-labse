from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments


def main():
    ds = load_dataset("adeshkin/khakas-russian-parallel-corpus", split="train")
    ds = ds.select_columns(["kjh", "ru"])
    ds = ds.train_test_split(test_size=0.01, seed=42)

    trans_eval = TranslationEvaluator(
        source_sentences=ds["test"]["kjh"],
        target_sentences=ds["test"]["ru"],
        name="kjh-ru-random",
        batch_size=8,
    )

    model = SentenceTransformer('sentence-transformers/LaBSE')

    train_loss = MultipleNegativesRankingLoss(model=model)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir='labse-kjh-ru-mnrl-1',
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=1000,
        fp16=True,
        learning_rate=2e-5,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        save_strategy="no",
        logging_steps=500,
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        loss=train_loss,
        evaluator=trans_eval,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == '__main__':
    main()
