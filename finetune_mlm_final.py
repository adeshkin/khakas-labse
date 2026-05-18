from datasets import load_dataset, Dataset
import os
import pandas as pd
import json
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, \
    Trainer, EarlyStoppingCallback
import random
from google.colab import userdata


def get_training_corpus(sentences, batch_size=1000):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i: i + batch_size]


def get_vocab2id(tkn_dir):
    tokenizer_json_path = os.path.join(tkn_dir, "tokenizer.json")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    vocab2id = tokenizer_data["model"]["vocab"]

    return vocab2id


def test_tokenizer(tkn_dir, all_kjh_sents, tokens_to_add, old_vocab2id):
    kjh_tokenizer = AutoTokenizer.from_pretrained(tkn_dir)
    kjh_vocab2id = get_vocab2id(tkn_dir)

    kjh_vocab = set(kjh_vocab2id.keys())
    old_vocab = set(old_vocab2id.keys())

    print(f"# old tokens: {len(old_vocab)}")
    print(f"# added tokens: {len(tokens_to_add)}")
    print(f"# new tokens: {len(kjh_vocab)}")

    assert len(kjh_tokenizer) == len(tokens_to_add) + len(old_vocab)
    assert len(kjh_vocab) == len(tokens_to_add) + len(old_vocab)
    assert tokens_to_add.issubset(kjh_vocab)
    assert old_vocab.issubset(kjh_vocab)
    assert tokens_to_add.union(old_vocab) == kjh_vocab

    for token, token_id in old_vocab2id.items():
        assert token_id == kjh_vocab2id[token]

    used_new_tokens = set()
    for k_sent in all_kjh_sents:
        tok_sent = kjh_tokenizer.tokenize(k_sent)
        assert '[UNK]' not in tok_sent

        input_ids = kjh_tokenizer.encode(k_sent)
        decoded = kjh_tokenizer.decode(input_ids, skip_special_tokens=True)
        assert len(decoded) > 0

        for tok in tok_sent:
            if tok in tokens_to_add:
                used_new_tokens.add(tok)

    assert len(used_new_tokens) > 0, "Новые токены не используются при токенизации!"
    print('test passed!')


def filter_func(examples):
    kjh_sent = examples["kjh"]
    return kjh_sent is not None and len(kjh_sent) >= 5


def prepare_mono_data():
    ds_para = load_dataset("adeshkin/khakas-russian-parallel-corpus", split="train")
    ds_mono = load_dataset("adeshkin/khakas-monolingual-corpus", split="train")

    ds_para = ds_para.filter(filter_func)
    ds_mono = ds_mono.filter(filter_func)

    ds_para = ds_para.select_columns(["kjh"])
    ds_mono = ds_mono.select_columns(["kjh"])

    df_para = ds_para.to_pandas()
    df_mono = ds_mono.to_pandas()

    df_merged = pd.merge(df_para, df_mono, on="kjh", how="outer")

    assert len(df_merged) == len(df_para) + len(df_mono)

    return df_merged


def update_tokenizer(all_kjh_sents, model_name,
                     tkn_tmp_dir='tokenizer_temp', tkn_dir='tokenizer_with_kjh'):
    assert not os.path.exists(tkn_tmp_dir)
    assert not os.path.exists(tkn_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tkn_tmp_dir)

    old_vocab = set(tokenizer.get_vocab().keys())

    khakas_tokenizer = tokenizer.train_new_from_iterator(
        get_training_corpus(all_kjh_sents),
        vocab_size=2 ** 14
    )

    khakas_vocab = set(khakas_tokenizer.get_vocab().keys())
    tokens_to_add = khakas_vocab.difference(old_vocab)

    print(f"old: {len(old_vocab)}")
    print(f"new: {len(tokens_to_add)}")

    tokenizer_json_path = os.path.join(tkn_tmp_dir, "tokenizer.json")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    vocab2id = tokenizer_data["model"]["vocab"].copy()
    len_vocab2id = len(vocab2id)
    current_max_id = max(vocab2id.values())

    for token in tokens_to_add:
        assert token not in vocab2id
        current_max_id += 1
        tokenizer_data["model"]["vocab"][token] = current_max_id

    assert len(vocab2id) == len_vocab2id

    tokenizer_json_path = os.path.join(tkn_tmp_dir, "tokenizer.json")
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    tokenizer_kjh = AutoTokenizer.from_pretrained(tkn_tmp_dir)
    tokenizer_kjh.save_pretrained(tkn_dir)

    test_tokenizer(tkn_dir, all_kjh_sents, tokens_to_add, vocab2id)

    tokenizer = AutoTokenizer.from_pretrained(tkn_dir)

    return tokenizer


def main():
    df_merged = prepare_mono_data()
    ds_merged = Dataset.from_pandas(df_merged)
    ds_merged = ds_merged.train_test_split(test_size=0.05, seed=42)

    all_kjh_sents = df_merged["kjh"].tolist()
    random.shuffle(all_kjh_sents)

    model_name = 'cointegrated/LaBSE-en-ru'

    tokenizer = update_tokenizer(all_kjh_sents, model_name)

    def preprocess_func(examples):
        return tokenizer(examples['kjh'], truncation=False, return_special_tokens_mask=True)

    tok_ds = ds_merged.map(
        preprocess_func,
        batched=True,
        num_proc=4,
        remove_columns=ds_merged["train"].column_names,
    )

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_dataset = tok_ds.map(group_texts, batched=True, num_proc=4)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)  # todo

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    old_tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeds = model.bert.embeddings.word_embeddings.weight.data

    added_vocab = set(tokenizer.get_vocab().keys()).difference(set(old_tokenizer.get_vocab().keys()))
    unk_id = old_tokenizer.unk_token_id
    for token in added_vocab:
        clean_token = token.replace("##", "")
        old_ids = old_tokenizer(clean_token, add_special_tokens=False).input_ids
        if len(old_ids) != 0 and unk_id not in old_ids:
            idx = tokenizer.convert_tokens_to_ids(token)
            embeds[idx] = embeds[old_ids].mean(0)

    training_args = TrainingArguments(
        output_dir="/content/drive/MyDrive/experiments/labse-en-ru_kjh-mlm",
        eval_strategy="steps",
        save_strategy="best",
        learning_rate=5e-5,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        fp16=True,
        logging_steps=500,
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3,
                                  early_stopping_threshold=0.1)
        ],
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nЗагрузка лучшей модели...")
        if trainer.state.best_model_checkpoint:
            trainer._load_best_model()  # Встроенный метод Trainer
        else:
            print("Ошибка: чекпоинты не найдены.")

    trainer.push_to_hub(
        token=userdata.get('WRITE_HF_TOKEN'),
    )


if __name__ == "__main__":
    main()
