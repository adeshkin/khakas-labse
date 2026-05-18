from datasets import load_dataset
import pandas as pd


def filter_func(examples):
    kjh_sent = examples["kjh"]
    return kjh_sent is not None and len(kjh_sent) >= 5


def prepare_mono_data():
    

    return df_merged
