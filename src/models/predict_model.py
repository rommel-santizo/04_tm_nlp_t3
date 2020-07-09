import logging
import pickle

from typing import List

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
# x = pd.DataFrame({'content': [1, 2, 3]})

other_texts = [
    ['goverment','public','computer', 'time', 'graph'],
    ['survey', 'response', 'eps'],
    ['human', 'system', 'computer']]


def predict_model(documents: List[List[str]]):
    # DICCIONARIO
    with open('../../models/id2word.pkl', 'rb') as input_file:
        id2word = pickle.load(input_file)
    logging.info('READ id2word.pkl')
    # CORPUS
    # with open('../../models/corpus.pkl', 'rb') as input_file:
    #     corpus = pickle.load(input_file)
    # logging.info('READ corpus.pkl')
    # MODELO LDA
    with open('../../models/lda_model.pkl', 'rb') as input_file:
        lda_model = pickle.load(input_file)
    logging.info('READ lda_model.pkl')

    other_corpus = [id2word.doc2bow(text) for text in documents]
    print(other_corpus)
    predictions = []
    for unseen_doc in other_corpus:
        # print(unseen_doc)
        vector = lda_model[unseen_doc]
        predictions.append(vector)
        # print(vector)
        # print(predictions)
    return predictions


def main():
    logging.info('INI main()')
    predictions = predict_model(other_texts)
    print(predictions)
    logging.info('FIN main()')


if __name__ == "__main__":
    main()
