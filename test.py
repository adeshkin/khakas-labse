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


if __name__ == '__main__':
    main()
