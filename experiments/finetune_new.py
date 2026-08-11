import pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator, MSEEvaluator, SequentialEvaluator
from sentence_transformers.losses import MSELoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from huggingface_hub import login


def main():
    login()
    data_dir = "/content/drive/MyDrive/article khakas-mt/labse"
    repo_name = "labse-en-ru-kjh340"

    raw_dataset = load_dataset("adeshkin/google-smol-en-ru-kjh", split="train")

    def clean_func(examples):
        kjh = examples["kjh"]
        trans = examples["Translation АНИСИМОВ"]
        return kjh is not None and trans is not None and len(kjh) > 4 and len(trans) > 4

    dataset = raw_dataset.filter(clean_func)
    dataset = dataset.rename_column("kjh", "anchor")
    dataset = dataset.rename_column("Translation АНИСИМОВ", "positive")
    dataset = dataset.select_columns(["anchor", "positive"])

    # --- ДОБАВЛЯЕМ ЛОГИКУ УЧИТЕЛЬ-УЧЕНИК ---
    model_name = "cointegrated/LaBSE-en-ru"

    # 1. Загружаем Учителя (он не будет обучаться, только генерировать таргеты)
    teacher_model = SentenceTransformer(model_name)

    # Функция для генерации эмбеддингов
    def add_teacher_embeddings(batch):
        # Учитель кодирует русский текст (positive).
        # Это будет "золотой стандарт" (label), к которому должен стремиться ученик.
        batch["label"] = teacher_model.encode(batch["positive"]).tolist()
        return batch

    print("Генерация векторов учителя...")
    # Применяем батчами для ускорения (занимает немного времени перед обучением)
    dataset = dataset.map(add_teacher_embeddings, batched=True, batch_size=64)
    # ----------------------------------------

    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # 2. Инициализируем Ученика (именно он будет обновлять веса)
    student_model = SentenceTransformer(model_name)

    # 3. Устанавливаем MSELoss для ученика
    train_loss = MSELoss(model=student_model)

    val_trans_evaluator = TranslationEvaluator(
        source_sentences=dataset["test"]["anchor"],   # хакасский
        target_sentences=dataset["test"]["positive"], # русский
        name="trans",
        batch_size=16
    )

    # 2. Второй эвалюатор: проверяет насколько MSE (ошибка) падает на тестовой выборке
    val_mse_evaluator = MSEEvaluator(
        source_sentences=dataset["test"]["positive"], # Русский текст идет Учителю
        target_sentences=dataset["test"]["anchor"],   # Хакасский текст идет Ученику
        teacher_model=teacher_model,                  # Передаем модель учителя!
        name="mse_distillation",
        batch_size=16
    )

    # 3. Объединяем их в один с помощью SequentialEvaluator
    combined_evaluator = SequentialEvaluator([val_trans_evaluator, val_mse_evaluator])
    # SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores))
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{data_dir}/{repo_name}",
        num_train_epochs=1,
        per_device_train_batch_size=16,  # Для MSE можно спокойно ставить 8, 16 или 32 (здесь не нужны огромные батчи!)
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
        model=student_model,  # Передаем ученика
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=train_loss,  # Передаем MSELoss
        evaluator=combined_evaluator,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()