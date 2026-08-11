


import random
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
import os
import logging
from warnings import simplefilter
import sys

from lingtrain_aligner import preprocessor, splitter, aligner, resolver, vis_helper, metrics

sns.set_theme(style="darkgrid")

# configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(process)d: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_labse.log", mode="a"),
    ],
)
logging.getLogger("matplotlib.font_manager").disabled = True
logger = logging.getLogger(__name__)


class ChainScoreEvaluator(SentenceEvaluator):
    """Evaluate a lingtrain chain score. This score calculates coefficient of unbrokenness."""

    def __init__(self, db_path, lang_from, lang_to, text1, text2, model, save_dir, evaluation_steps=100):
        super().__init__()
        self.db_path = db_path
        self.lang_from = lang_from
        self.lang_to = lang_to
        self.text1 = text1
        self.text2 = text2
        self.scores_1 = []
        self.scores_2 = []
        self.best_score = 0.0
        self.model = model
        self.evaluation_steps = evaluation_steps
        self.save_dir = save_dir

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        clear_output(wait=True)

        text1_prepared = preprocessor.mark_paragraphs(self.text1)
        text2_prepared = preprocessor.mark_paragraphs(self.text2)

        splitted_from = splitter.split_by_sentences_wrapper(text1_prepared, self.lang_from)
        splitted_to = splitter.split_by_sentences_wrapper(text2_prepared, self.lang_to)

        logger.info(f'Splitted length: {self.lang_from} — {len(splitted_from)}, {self.lang_to} — {len(splitted_to)}')

        if os.path.isfile(self.db_path):
            os.unlink(self.db_path)
        aligner.fill_db(self.db_path, self.lang_from, self.lang_to, splitted_from, splitted_to)

        batch_ids = range(0, 1)
        aligner.align_db(self.db_path,
                         model_name='_',
                         batch_size=500,
                         window=50,  # tweak this parameter if needed
                         batch_ids=batch_ids,
                         save_pic=False,
                         embed_batch_size=100,
                         normalize_embeddings=True,
                         show_progress_bar=False,
                         shift=0,  # tweak this parameter if needed
                         model=self.model
                         )

        conflicts, rest = resolver.get_all_conflicts(self.db_path, min_chain_length=2, max_conflicts_len=6, batch_id=-1)
        logger.info(
            f'Resolving short conflicts for proper validation: {len(conflicts)} conflicts detected (min chains len are from 2 to 6).')
        resolver.resolve_all_conflicts(self.db_path, conflicts, model_name="_", show_logs=False, model=self.model)
        score_1 = metrics.f(self.db_path)
        score_2 = metrics.chain_score(self.db_path, mode='both')
        logger.info(f"Epoch: {epoch} steps: {steps}.")

        if score_1 > self.best_score:
            self.best_score = score_1
            logger.info(f"Score 1: {score_1} new best score (#1).")
            if self.best_score > 0.06:
                logger.info("Saving the model...")
                model.save(f'{self.save_dir}/best_model_{self.lang_from}_{self.lang_to}')
                logger.info(f"Model saved to {self.save_dir}/best_model_{self.lang_from}_{self.lang_to}.")
        else:
            logger.info(f"score 1: {score_1}")

        logger.info(f"score 2: {score_2}")
        self.scores_1.append(score_1)
        self.scores_2.append(score_2)

        vis_helper.visualize_alignment_by_db(self.db_path,
                                             output_path=f"{self.save_dir}/alignment_vis_{self.lang_from}_{self.lang_to}.png",
                                             batch_size=400,
                                             size=(600, 600),
                                             lang_name_from=self.lang_from,
                                             lang_name_to=self.lang_to,
                                             plt_show=True)

        if steps % self.evaluation_steps == 0:
            sns.lineplot(data=self.scores_1)

        return score_1


def main():
    LANG_1 = 'kjh'
    LANG_2 = 'ru'
    data_dir = '/content/drive/MyDrive/article khakas-mt/labse/data'
    path = f'{data_dir}/para_kjh_ru.csv'
    df = pd.read_csv(path)
    pairs = df[[LANG_1, LANG_2]].values.tolist()
    random.shuffle(pairs)

    train_dataset_orig = []
    for pair in pairs:
        train_dataset_orig.append({LANG_1: pair[0], LANG_2: pair[1]})

    model = SentenceTransformer('LaBSE')
    exp_name = 'finetune_labse_lingtrain'
    save_dir = f'./artifacts/{exp_name}'
    os.makedirs(save_dir, exist_ok=False)

    with open(f'{data_dir}/vasyutkino_ozero/{LANG_1}.txt', "r", encoding="utf8") as input1:
        text1 = input1.readlines()
    with open(f'{data_dir}/vasyutkino_ozero/{LANG_2}.txt', "r", encoding="utf8") as input2:
        text2 = input2.readlines()

    db_path = f"{save_dir}/alignment_{LANG_1}_{LANG_2}.db"
    evaluation_steps = 100

    evaluator = ChainScoreEvaluator(db_path, LANG_1, LANG_2, text1, text2, model, save_dir,
                                    evaluation_steps=evaluation_steps)

    num_epochs = 1
    train_batch_size = 8
    warmup_steps = 1000

    train_examples = [InputExample(texts=[x[LANG_1], x[LANG_2]], label=1.0) for x in train_dataset_orig]
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        save_best_model=False,
        use_amp=True,
        warmup_steps=warmup_steps,
        scheduler='warmupcosine',
        optimizer_params={'lr': 2e-5},  # 3e-4
    )



if __name__ == '__main__':
    main()
