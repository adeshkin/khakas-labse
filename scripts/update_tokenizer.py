import os
import pandas as pd
import random
import json
from transformers import AutoTokenizer

from preprocess_text import preproc


def prepare_mono_text(mono_path, para_path, mono_col_name):
    df_mono = pd.read_csv(mono_path)
    df_para = pd.read_csv(para_path)

    all_texts = df_para[mono_col_name].tolist() + df_mono[mono_col_name].tolist()
    all_texts_norm = [preproc(t) for t in all_texts]
    print(f'# texts: {len(all_texts_norm)}')

    return all_texts_norm


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


def main():
    data_dir = '/home/adeshkin/khakas_projects/khakas-mt/data/final'
    mono_path = f'{data_dir}/mono_kjh.csv'
    para_path = f'{data_dir}/para_kjh_ru.csv'
    mono_col_name = 'kjh'

    model_name = 'cointegrated/LaBSE-en-ru'

    tkn_tmp_dir = './artifacts/tokenizer_temp'
    assert not os.path.exists(tkn_tmp_dir)

    tkn_dir = './artifacts/tokenizer_with_kjh'
    assert not os.path.exists(tkn_dir)

    all_kjh_sents = prepare_mono_text(mono_path, para_path, mono_col_name)
    random.shuffle(all_kjh_sents)

    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
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


if __name__ == "__main__":
    main()
