import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer
from dotenv import load_dotenv

load_dotenv()


def filter_func(examples):
    kjh_sent = examples["kjh"]
    return kjh_sent is not None and len(kjh_sent) > 4


def main():
    model_name = "sentence-transformers/LaBSE"
    output_dir = "/content/drive/MyDrive/article khakas-mt/labse-khakas-mlm"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/LaBSE")

    raw_dataset1 = load_dataset("adeshkin/khakas-russian-parallel-corpus", split="train", token=os.getenv("TOKEN"))
    raw_dataset2 = load_dataset("adeshkin/khakas-monolingual-corpus", split="train")

    dataset = raw_dataset.filter(clean_func)
    dataset = dataset.select_columns(["kjh"])

    # 3. Токенизация и нарезка текста на чанки (блоки)
    chunk_size = 128  # Длина последовательности (в токенах)

    def tokenize_function(examples):
        # Токенизируем без паддинга (сделаем его позже)
        return tokenizer(examples["kjh"], truncation=False, return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["kjh"])

    def group_texts(examples):
        # Склеиваем все тексты в один длинный массив
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Отрезаем остаток, который не делится на chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # Нарезаем на блоки по chunk_size (128 токенов)
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Нарезка текстов на блоки для MLM...")
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    # Разбиваем на train/test, чтобы следить за тем, не переобучились ли мы
    split_dataset = lm_dataset.train_test_split(test_size=0.05, seed=42)

    # 4. Data Collator (он автоматически будет маскировать 15% слов прямо во время обучения)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Стандартное значение из статьи про BERT
    )

    # 5. Настройки обучения (Гиперпараметры)
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",  # Проверяем ошибку в конце каждой эпохи
        save_strategy="epoch",
        learning_rate=5e-5,  # Для MLM обычно берут rate чуть выше (5e-5 или 1e-4)
        num_train_epochs=10,  # Количество эпох (см. пояснение ниже)
        per_device_train_batch_size=32,  # На T4/L4 в Colab 32-64 должно влезть
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        fp16=True,  # Ускоряет обучение на GPU
        logging_steps=50,
        report_to="none",
        push_to_hub=False  # На хаб пока не пушим, это промежуточная модель
    )

    # 6. Запуск обучения
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    print("Начинаем обучение MLM...")
    trainer.train()

    # 7. Сохраняем финальную модель и токенизатор
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    tokenizer.save_pretrained(output_dir)
    print(f"Модель сохранена в {output_dir}")


if __name__ == "__main__":
    main()
