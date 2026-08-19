# khakas-sent-emb

> **Обученная модель:** [adeshkin/labse-kjh-ru-mnrl-1](https://huggingface.co/adeshkin/labse-kjh-ru-mnrl-1)  
> Модель обучена по скрипту `experiments/finetune_mnrl_1.py`.

Обучение sentence-эмбеддингов для хакасского языка (kjh) на базе [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) / [LaBSE-en-ru](https://huggingface.co/cointegrated/LaBSE-en-ru).

## Идея пайплайна

1. **Подготовка данных** — очистка текста, выравнивание параллельных предложений (kjh↔ru), сбор моноязычного корпуса.
2. **Расширение токенизатора** — добавление хакасских токенов, которых не было в исходной модели.
3. **MLM (Task-Adaptive Pretraining)** — доучивание модели на хакасских текстах (маскированное языковое моделирование), чтобы модель освоила лексику и морфологию языка *до* выравнивания эмбеддингов.
4. **Teacher-Student (MSE Loss)** — выравнивание хакасских векторов с векторами учителя (LaBSE-en-ru) на параллельном корпусе.
5. **Contrastive fine-tuning (MNRL)** — финальная «полировка» точности на `MultipleNegativesRankingLoss`, небольшой LR, максимально большой батч.
6. **Публикация** — загрузка обученной модели на HuggingFace Hub.

| Этап | Данные | Эпохи | LR | Batch |
| :--- | :--- | :--- | :--- | :--- |
| MLM | 250к предложений | 5–10 | 5e-5 | 32–64 |
| MSE (teacher-student) | 150к пар | 3–5 | 2e-5 | 16–32 |
| MNRL | 150к пар | 1 | 1e-6 | 32+ |

## Структура репозитория

```
scripts/       — основной пайплайн
  preprocess_text.py   очистка/нормализация текста
  prepare_mono.py       подготовка моноязычного корпуса
  check.py               проверка выравнивания пар предложений (lingtrain_aligner)
  update_tokenizer.py    расширение токенизатора хакасскими токенами
  train.py               MLM-дообучение + teacher-student выравнивание
  test.py                проверка модели (маскирование, инспекция токенизации)
  push_to_hf.py          публикация модели на HuggingFace Hub

experiments/    — альтернативные варианты/итерации finetune-этапа (черновики, не основной путь)
  finetune.py, finetune_new.py, finetune_mnrl_1.py,
  finetune_mlm.py, finetune_mlm_final.py,
  finetune_labse_lingtrain.py, make_multilingual.py
```

## Установка

```bash
pip install -r requirements.txt
```

`requirements_old.txt` — зафиксированные версии зависимостей из более раннего рабочего окружения (на случай проблем совместимости).
