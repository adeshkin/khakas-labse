import pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from huggingface_hub import login


def main():
    login()
    data_dir = "/content/drive/MyDrive/article khakas-mt/labse"
    repo_name = "labse-en-ru-kjh340"

    # df = load_dataset("adeshkin/google-smol-en-ru-kjh", split="train").to_pandas()
    # df = df.dropna()
    # df = df[df["kjh"].str.len() > 4]
    # df = df[df["Translation АНИСИМОВ"].str.len() > 4]

    # dataset = Dataset.from_dict({
    #     "anchor": df["kjh"].tolist(),
    #     "positive": df["Translation АНИСИМОВ"].tolist()
    # })
    raw_dataset = load_dataset("adeshkin/google-smol-en-ru-kjh", split="train")

    def clean_func(examples):
        # Проверка на None и длину строк
        kjh = examples["kjh"]
        trans = examples["Translation АНИСИМОВ"]
        return kjh is not None and trans is not None and len(kjh) > 4 and len(trans) > 4

    dataset = raw_dataset.filter(clean_func)

    # Переименовываем колонки под стандарт SentenceTransformers (anchor, positive)
    dataset = dataset.rename_column("kjh", "anchor")
    dataset = dataset.rename_column("Translation АНИСИМОВ", "positive")
    dataset = dataset.select_columns(["anchor", "positive"])
    print(dataset.column_names)

    # Разбиваем на train и test
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    model_name = "cointegrated/LaBSE-en-ru"
    model = SentenceTransformer(model_name)

    train_loss = MultipleNegativesRankingLoss(model=model)
    val_evaluator = TranslationEvaluator(
        source_sentences=dataset["test"]["anchor"],
        target_sentences=dataset["test"]["positive"],
        name="trans",       # префикс, который будет добавлен к метрикам в логах
        batch_size=16
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=f"{data_dir}/{repo_name}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0.1,
        eval_strategy="steps",
        logging_steps=40,
        save_strategy="no",
        fp16=True,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=train_loss,
        evaluator=val_evaluator,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
