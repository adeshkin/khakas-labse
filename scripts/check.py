from lingtrain_aligner import preprocessor, splitter
from preprocess_text import preproc

def main():
    data_dir = '/home/adeshkin/Downloads/vasyutkino_ozero'
    lang_from = 'kjh'
    lang_to = 'ru'

    with open(f'{data_dir}/{lang_from}_fixed.txt', "r", encoding="utf8") as input1:
        text1 = input1.readlines()

    with open(f'{data_dir}/{lang_to}_fixed.txt', "r", encoding="utf8") as input2:
        text2 = input2.readlines()

    print(len(text1), len(text2))
    print()

    text1_prepared = preprocessor.mark_paragraphs(text1)
    text2_prepared = preprocessor.mark_paragraphs(text2)
    print(len(text1_prepared), len(text2_prepared))
    print()

    splitted_from = splitter.split_by_sentences_wrapper(text1_prepared, lang_from)
    splitted_to = splitter.split_by_sentences_wrapper(text2_prepared, lang_to)
    print(len(splitted_from), len(splitted_to))
    print()


if __name__ == '__main__':
    main()
