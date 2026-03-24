import random
import torch
import gc
import pandas as pd
from datetime import datetime
from transformers.optimization import Adafactor
from transformers import AutoTokenizer, AutoModelForPreTraining
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from preprocess_text import preproc


def prepare_pairs(para_path, columns):
    df_para = pd.read_csv(para_path)
    for col in columns:
        df_para[col] = df_para[col].apply(preproc)
    all_pairs = df_para[columns].values.tolist()

    return all_pairs


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def get_acc(embeddings):
    batch_size = embeddings.shape[0] // 2
    with torch.no_grad():
        scores = torch.matmul(
            embeddings[:batch_size].detach(),
            embeddings[batch_size:].T
        ).cpu().numpy()
    a1 = (scores.argmax(1) == np.arange(batch_size)).mean()
    a2 = (scores.argmax(0) == np.arange(batch_size)).mean()
    return (a1 + a2) / 2


def get_contrastive_loss(embs1, embs2, loss_fn, margin=0.3):
    bs = embs1.shape[0]
    d = embs1.device
    embs1 = torch.nn.functional.normalize(embs1)
    embs2 = torch.nn.functional.normalize(embs2)
    all_scores = torch.matmul(embs1, embs2.T)
    if margin:
        all_scores = all_scores - torch.eye(bs, device=d) * margin
    diag_ids = torch.arange(bs, device=d)

    return loss_fn(all_scores, diag_ids) + loss_fn(all_scores.T, diag_ids)


def prepare_model(model_name, tokenizer):
    model = AutoModelForPreTraining.from_pretrained(model_name)
    old_tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))
    embeds = model.bert.embeddings.word_embeddings.weight.data

    added_vocab = set(tokenizer.get_vocab()).difference(set(old_tokenizer.get_vocab()))
    unk_count = 0
    for token in added_vocab:
        clean_token = token.replace("##", "")
        old_ids = old_tokenizer(clean_token, add_special_tokens=False).input_ids
        if len(old_ids) == 0:
            old_ids = [old_tokenizer.unk_token_id]
            unk_count += 1

        idx = tokenizer.convert_tokens_to_ids(token)
        embeds[idx] = embeds[old_ids].mean(0)

    print('all', len(added_vocab))
    print('unk_count', unk_count)

    return model


def main():
    cleanup()
    model_name = 'cointegrated/LaBSE-en-ru'
    art_dir = './artifacts'
    tkn_dir = f'{art_dir}/tokenizer_with_kjh'
    para_path = '/home/adeshkin/khakas_projects/khakas-mt/data/final/para_kjh_ru.csv'
    langs = ['kjh', 'ru']
    log_dir = f'{art_dir}/logs'
    os.makedirs(log_dir, exist_ok=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"labse_finetune_kjh_ru_{timestamp}"
    writer = SummaryWriter(os.path.join(log_dir, exp_name))

    model_dir = f"{art_dir}/model_checkpoints/{exp_name}"
    os.makedirs(model_dir, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(tkn_dir)
    model = prepare_model(model_name, tokenizer)

    all_pairs = prepare_pairs(para_path, langs)
    batch_size = 4
    margin = 0.3
    num_steps = 500_000

    model.cuda()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.bert.embeddings.word_embeddings.parameters():
        p.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False, relative_step=False, lr=1e-5, clip_threshold=1.0
    )

    losses = []
    accuracies = []
    model.train()
    for i in range(num_steps):
        kjh_examples, ru_examples = [list(p) for p in zip(*random.choices(all_pairs, k=batch_size))]
        try:
            batch = tokenizer(ru_examples + kjh_examples,
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=128).to(model.device)
            out = model.bert(**batch, output_hidden_states=True)
            embeddings = torch.nn.functional.normalize(out.pooler_output)
            embs1 = embeddings[:batch_size].detach()  # keep Russian embeddings frozen
            embs2 = embeddings[batch_size:]  # update Khakas embeddings

            loss = get_contrastive_loss(embs1, embs2, loss_fn, margin=margin)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            acc_value = get_acc(embeddings)
            accuracies.append(acc_value)
        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
            batch, out, embeddings, all_scores, loss = None, None, None, None, None
            cleanup()
            print('error', max(len(s) for s in kjh_examples + ru_examples))
            continue

        except KeyboardInterrupt as e:
            print('\nKeyboardInterrupt detected.')
            break

        if i % 100 == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/loss_mean_100", np.mean(losses[-100:]), i)
            writer.add_scalar("train/accuracy", acc_value, i)
            writer.add_scalar("train/accuracy_mean_100", np.mean(accuracies[-100:]), i)
            writer.flush()

            print(f"Step {i} | Loss: {np.mean(losses[-100:]):.4f} (Acc: {np.mean(accuracies[-100:]):.4f})\n")
            last_model_path = f"{model_dir}/last-{i}"
            model.save_pretrained(last_model_path)

    writer.close()


if __name__ == '__main__':
    main()
