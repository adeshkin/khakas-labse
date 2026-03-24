import random
import torch
import gc
import pandas as pd
from datetime import datetime
from transformers.optimization import Adafactor
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling
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


# def get_acc(embeddings):
#     batch_size = embeddings.shape[0] // 2
#     with torch.no_grad():
#         scores = torch.matmul(
#             embeddings[:batch_size].detach(),
#             embeddings[batch_size:].T
#         ).cpu().numpy()
#     a1 = (scores.argmax(1) == np.arange(batch_size)).mean()
#     a2 = (scores.argmax(0) == np.arange(batch_size)).mean()
#     return (a1 + a2) / 2

def get_acc(embs1, embs2):
    batch_size = embs1.shape[0]
    with torch.no_grad():
        scores = torch.matmul(embs1, embs2.T).cpu().numpy()
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


def train_v0():
    cleanup()
    model_name = 'cointegrated/LaBSE-en-ru'
    art_dir = './artifacts'
    tkn_dir = f'{art_dir}/tokenizer_with_kjh'
    para_path = '/home/adeshkin/khakas_projects/khakas-mt/data/final/para_kjh_ru.csv'
    langs = ['kjh', 'ru']
    log_dir = f'{art_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"labse_finetune_kjh_ru_{timestamp}"
    writer = SummaryWriter(os.path.join(log_dir, exp_name))

    model_dir = f"{art_dir}/model_checkpoints/{exp_name}"
    os.makedirs(model_dir, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(tkn_dir)
    model = prepare_model(model_name, tokenizer)

    all_pairs = prepare_pairs(para_path, langs)
    batch_size = 32
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
            # batch = tokenizer(ru_examples + kjh_examples,
            #                   return_tensors='pt',
            #                   padding=True,
            #                   truncation=True,
            #                   max_length=128).to(model.device)
            # out = model.bert(**batch, output_hidden_states=True)
            # embeddings = torch.nn.functional.normalize(out.pooler_output)
            # embs1 = embeddings[:batch_size].detach()  # keep Russian embeddings frozen
            # embs2 = embeddings[batch_size:]  # update Khakas embeddings
            # 1. Русские примеры прогоняем без градиентов (экономит ~50% памяти)
            with torch.no_grad():
                ru_batch = tokenizer(ru_examples,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True,
                                     max_length=128).to(model.device)
                ru_out = model.bert(
                    **ru_batch)  # output_hidden_states=True здесь не обязательно, если используете pooler_output
                embs1 = torch.nn.functional.normalize(ru_out.pooler_output).detach()

            # 2. Хакасские примеры прогоняем с вычислением градиентов
            kjh_batch = tokenizer(kjh_examples,
                                  return_tensors='pt',
                                  padding=True,
                                  truncation=True,
                                  max_length=128).to(model.device)
            kjh_out = model.bert(**kjh_batch)
            embs2 = torch.nn.functional.normalize(kjh_out.pooler_output)

            loss = get_contrastive_loss(embs1, embs2, loss_fn, margin=margin)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            # accuracies.append(get_acc(embeddings))
            accuracies.append(get_acc(embs1, embs2))
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
            writer.add_scalar("train/loss_mean_1000", np.mean(losses[-1000:]), i)
            writer.add_scalar("train/accuracy_mean_1000", np.mean(accuracies[-1000:]), i)
            writer.flush()

            print(f"Step {i} | Loss: {np.mean(losses[-1000:]):.4f} | Accuracy: {np.mean(accuracies[-1000:]):.4f}\n")

    model_path = f"{model_dir}/labse_kjh_ru_v0"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    writer.close()


def get_acc2(e1, e2):
    batch_size = e1.shape[0]
    with torch.no_grad():
        scores = torch.matmul(e1, e2.T).cpu().numpy()
    a1 = (scores.argmax(1) == np.arange(batch_size)).mean()
    a2 = (scores.argmax(0) == np.arange(batch_size)).mean()
    return (a1 + a2) / 2


def corrupt_sentence(sent, ix, all_pairs, p_edit=0.5):
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


def corrupt_pair(pair, all_pairs):
    """ Corrupt one (randomly chosen) sentence in a pair """
    pair = list(pair)
    ix = random.choice([0, 1])
    sent = pair[ix]
    pair[ix] = corrupt_sentence(sent, ix, all_pairs)
    return pair


def get_pairs_batch(short_pairs, bs=4):
    pp = random.choices(short_pairs, k=int(np.ceil(bs / 2)))
    labels = [1] * len(pp) + [0] * len(pp)
    if random.random() < 0.5:
        # make negatives by swapping sentence with a random one
        pp.extend([(pp[i][0], pp[i - 1][1]) for i in range(len(pp))])
    else:
        # make negatives by corrupting existing sentences
        pp.extend([corrupt_pair(p, short_pairs) for p in pp])
    pp = [[x, y] if random.random() < 0.5 else [y, x] for x, y in pp]

    return [list(t) for t in zip(*pp)], labels


def train_v1():
    art_dir = './artifacts'
    tkn_dir = f'{art_dir}/tokenizer_with_kjh'
    para_path = '/home/adeshkin/khakas_projects/khakas-mt/data/final/para_kjh_ru.csv'
    base_model_path = './artifacts/model_checkpoints/labse_finetune_kjh_ru_20260324_120536/labse_kjh_ru_v0'

    langs = ['kjh', 'ru']
    log_dir = f'{art_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"labse_finetune_kjh_ru_{timestamp}"
    writer = SummaryWriter(os.path.join(log_dir, exp_name))

    model_dir = f"{art_dir}/model_checkpoints/{exp_name}"
    os.makedirs(model_dir, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(tkn_dir)

    all_pairs = prepare_pairs(para_path, langs)
    short_pairs = [p for p in tqdm(all_pairs) if len(tokenizer.encode(*p)) <= 100]
    all_kjh_sents = [p[0] for p in all_pairs]
    model = AutoModelForPreTraining.from_pretrained(base_model_path).cuda()

    teacher_model_name = 'cointegrated/LaBSE-en-ru'
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModel.from_pretrained(teacher_model_name).cuda()

    # collator = DataCollatorForWholeWordMask(tokenizer, mlm=True, mlm_probability=0.3)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, whole_word_mask=True, mlm_probability=0.3)
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
            sents = random.choices(all_kjh_sents, k=mlm_batch_size)
            # 1. Tokenize EVERYTHING at once.
            # This creates the 'offset_mapping' as a [2, N, 2] tensor immediately.
            encodings = tokenizer(
                sents,
                padding=True,
                truncation=True,
                max_length=128,
                return_offsets_mapping=True,
                return_tensors="pt"  # This is the critical part
            )

            # 2. Convert the batch of tensors into a list of dicts (features)
            # Each 'item' will contain 1D tensors (e.g., input_ids is [N], offset_mapping is [N, 2])
            features = []
            for j in range(len(sents)):
                item = {key: val[j] for key, val in encodings.items()}
                features.append(item)

            # 3. Pass to collator.
            # Because the features are already tensors, the collator will stack them
            # into a 3D tensor perfectly, and 'offsets[:, :, 0]' will work.
            kjh_batch = {k: v.to(model.device) for k, v in collator(features).items()}

            if (kjh_batch['labels'] != -100).any():
                loss = loss_fn(
                    model(**kjh_batch).prediction_logits.view(-1, model.config.vocab_size),
                    kjh_batch['labels'].view(-1)
                )
                # Дополнительная проверка на аномальные значения
                if not torch.isnan(loss):
                    loss.backward()
                    losses_mlm.append(loss.item())
                else:
                    print("MLM loss is NaN, skipping backward")
            else:
                print("No tokens masked in this batch, skipping step")

            # cross-encoder step
            pp, pl = get_pairs_batch(short_pairs, bs=4)
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

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            batch, out, embeddings, all_scores, loss = None, None, None, None, None
            cleanup()
            print('error', max(len(s) for s in kjh_examples + ru_examples))
            continue

        except KeyboardInterrupt as e:
            print('\nKeyboardInterrupt detected.')
            break

        if i % 100 == 0:
            writer.add_scalar("train/loss_mean_100", np.mean(losses2[-100:]), i)
            writer.add_scalar("train/accuracy_mean_100", np.mean(accuracies2[-100:]), i)
            writer.add_scalar("train/loss_mlm_mean_100", np.mean(losses_mlm[-100:]), i)
            writer.add_scalar("train/loss_ce_mean_100", np.mean(losses_ce[-100:]), i)
            writer.flush()

            print(f"Step {i} | Loss: {np.mean(losses2[-100:]):.4f} | Accuracy: {np.mean(accuracies2[-100:]):.4f}"
                  f" | MLM Loss: {np.mean(losses_mlm[-100:]):.4f} | CE Loss: {np.mean(losses_ce[-100:]):.4f}\n")

    model_path = f"{model_dir}/labse_kjh_ru_v1"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    writer.close()


if __name__ == '__main__':
    train_v0()
