import random
import torch
import gc
import pandas as pd
from datetime import datetime
from transformers.optimization import Adafactor
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel, DataCollatorForWholeWordMask
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm.auto import tqdm, trange

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

            loss_value = loss.item()
            acc_value = get_acc(embeddings)

            losses.append(loss_value)
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

        if i % 1000 == 0 and i > 0:
            writer.add_scalar("train/loss", loss_value, i)
            writer.add_scalar("train/loss_mean_100", np.mean(losses[-1000:]), i)
            writer.add_scalar("train/accuracy", acc_value, i)
            writer.add_scalar("train/accuracy_mean_100", np.mean(accuracies[-1000:]), i)
            writer.flush()

            print(f"Step {i} | Loss: {np.mean(losses[-1000:]):.4f} | Accuracy: {np.mean(accuracies[-1000:]):.4f}\n")

    model_path = f"{model_dir}/labse_kjh_ru_v0"
    model.save_pretrained(model_path)
    writer.close()


def get_acc2(e1, e2):
    batch_size = e1.shape[0]
    with torch.no_grad():
        scores = torch.matmul(e1, e2.T).cpu().numpy()
    a1 = (scores.argmax(1) == np.arange(batch_size)).mean()
    a2 = (scores.argmax(0) == np.arange(batch_size)).mean()
    return (a1 + a2) / 2


def corrupt_sentence(sent, ix, p_edit=0.5):
    sent = sent.split()
    old_sent = sent[:]
    while sent == old_sent:
        # insert a random word
        if random.random() < p_edit or len(sent) == 1:
            other_sent = random.choice(all_pairs)[ix].split()
            sent.insert(random.randint(0, len(sent) - 1), random.choice(other_sent))
        # replace a random word
        if random.random() < p_edit and len(sent) > 1:
            other_sent = random.choice(all_pairs)[ix].split()
            sent[random.randint(0, len(sent) - 1)] = random.choice(other_sent)
        # remove a word
        if random.random() < p_edit and len(sent) > 1:
            sent.pop(random.randint(0, len(sent) - 1))
        # swap words
        if random.random() < p_edit and len(sent) > 1:
            i, j = random.sample(range(len(sent)), 2)
            sent[i], sent[j] = sent[j], sent[i]
    return ' '.join(sent)


def corrupt_pair(pair):
    """ Corrupt one (randomly chosen) sentence in a pair """
    pair = list(pair)
    ix = random.choice([0, 1])
    sent = pair[ix]
    pair[ix] = corrupt_sentence(sent, ix)
    return pair


def get_pairs_batch(bs=4):
    pp = random.choices(short_pairs, k=int(np.ceil(bs / 2)))
    labels = [1] * len(pp) + [0] * len(pp)
    if random.random() < 0.5:
        # make negatives by swapping sentence with a random one
        pp.extend([(pp[i][0], pp[i - 1][1]) for i in range(len(pp))])
    else:
        # make negatives by corrupting existing sentences
        pp.extend([corrupt_pair(p) for p in pp])
    pp = [[x, y] if random.random() < 0.5 else [y, x] for x, y in pp]

    return [list(t) for t in zip(*pp)], labels


def train1():
    base_model_name = 'labse_erzya_v0'
    all_pairs = []
    model = AutoModelForPreTraining.from_pretrained(base_model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    pair_lens = [len(tokenizer.encode(*p)) for p in tqdm(all_pairs)]
    print(pd.Series(pair_lens).quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1]))
    short_pairs = [p for p in tqdm(all_pairs) if len(tokenizer.encode(*p)) <= 100]
    print(len(all_pairs), len(short_pairs))

    teacher_model_name = 'cointegrated/LaBSE-en-ru'
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModel.from_pretrained(teacher_model_name).cuda()

    collator = DataCollatorForWholeWordMask(tokenizer, mlm=True, mlm_probability=0.3)
    for p in model.parameters():
        p.requires_grad = True
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False, relative_step=False,
        lr=2e-6,  # make it very slow, because we want to update too many parameters
        clip_threshold=1.0
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    mlm_batch_size = 2
    batch_size = 4
    margin = 0.3
    losses2 = []
    accuracies2 = []
    losses_mlm = []
    losses_ce = []
    model.train()
    tq = trange(300_000)
    for i in tq:
        kjh_examples, ru_examples = [list(p) for p in zip(*random.choices(all_pairs, k=batch_size))]
        try:
            # translation ranking step
            # in half cases, pull embeddings to the teacher; in other half - to self.
            tm, tt = (teacher_model, teacher_tokenizer) if random.random() < 0.5 else (model.bert, tokenizer)
            ru_batch = tt(ru_examples, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                ru_emb = torch.nn.functional.normalize(tm(**ru_batch.to(teacher_model.device)).pooler_output)

            kjh_batch = tokenizer(kjh_examples, return_tensors='pt', padding=True, truncation=True, max_length=128)
            kjh_emb = torch.nn.functional.normalize(model.bert(**kjh_batch.to(model.device)).pooler_output)

            loss = get_contrastive_loss(ru_emb, kjh_emb, loss_fn, margin=margin)
            loss.backward()
            losses2.append(loss.item())
            accuracies2.append(get_acc2(kjh_emb, ru_emb))

            # mlm step
            sents = random.choices(all_sentences, k=mlm_batch_size)
            kjh_batch = {k: v.to(model.device) for k, v in collator([tokenizer(s) for s in sents]).items()}
            loss = loss_fn(
                model(**kjh_batch).prediction_logits.view(-1, model.config.vocab_size),
                kjh_batch['labels'].view(-1)
            )
            loss.backward()
            losses_mlm.append(loss.item())

            # cross-encoder step
            pp, pl = get_pairs_batch(bs=4)
            loss = loss_fn(
                model(
                    **tokenizer(*pp, padding=True, truncation=True, max_length=128, return_tensors='pt').to(
                        model.device)
                ).seq_relationship_logits.view(-1, 2),
                torch.tensor(pl, device=model.device)
            )
            loss.backward()
            losses_ce.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
            batch, out, embeddings, all_scores, loss = None, None, None, None, None
            cleanup()
            print('error', max(len(s) for s in kjh_examples + ru_examples))
            continue
        if i % 100 == 0:
            print(i, np.mean(losses2[-100:]), np.mean(accuracies2[-100:]), np.mean(losses_mlm[-100:]),
                  np.mean(losses_ce[-100:]))



if __name__ == '__main__':
    main()
