import torch
from transformers import AutoTokenizer, AutoModelForPreTraining


def test_mask(text, tokenizer, model):
    input = tokenizer(text, return_tensors="pt").to(model.device)
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)[0]

    with torch.no_grad():
        output = model(**input)
        logits = output.prediction_logits
        softmax = torch.softmax(logits, dim=-1)
        mask_word = softmax[0, mask_index, :]
        top_10 = torch.topk(mask_word, 10, dim=1)[1][0]
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)


def main():
    art_dir = './artifacts'
    tkn_dir = f'{art_dir}/tokenizer_with_kjh'
    base_model_path = '/home/adeshkin/khakas_projects/khakas-sent-emb/artifacts/model_checkpoints/labse_finetune_kjh_ru_20260324_162659/labse_kjh_ru_v1'
    model = AutoModelForPreTraining.from_pretrained(base_model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(tkn_dir)
    text = 'Олғаннар [MASK] хығырчалар.'
    test_mask(text, tokenizer, model)

from transformers import DataCollatorForLanguageModeling
from train import prepare_pairs
import random

def example():
    art_dir = './artifacts'
    tkn_dir = f'{art_dir}/tokenizer_with_kjh'
    tokenizer = AutoTokenizer.from_pretrained(tkn_dir)
    mlm_probability = 0.15  # 0.3
    para_path = '/home/adeshkin/khakas_projects/khakas-mt/data/final/para_kjh_ru.csv'
    langs = ['kjh', 'ru']
    all_pairs = prepare_pairs(para_path, langs)
    all_kjh_sents = [p[0] for p in all_pairs]
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, whole_word_mask=True,
                                               mlm_probability=mlm_probability)

    sents = random.choices(all_kjh_sents, k=4)

    encodings = tokenizer(
        sents,
        padding=True,
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
        return_tensors="pt"  # This is the critical part
    )

    features = []
    for j in range(len(sents)):
        item = {key: val[j] for key, val in encodings.items()}
        features.append(item)

    kjh_batch = {k: v for k, v in collator(features).items()}
    for i in range(len(sents)):
        print(tokenizer.tokenize(sents[i]))
        tokens = tokenizer.convert_ids_to_tokens(kjh_batch['input_ids'][i])
        print(tokens)
        print(tokens.count('[MASK]'), len(tokens), tokens.count('[MASK]')/len(tokens))
        print()


if __name__ == '__main__':
    example()
