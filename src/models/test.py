import logging

from src.models.predict_model import predict_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
# x = pd.DataFrame({'content': [1, 2, 3]})

other_texts = [
    ['This goverment made public access to computer'],
    ['Please make a survey to response for eps']]


def main():
    logging.info('INI main()')
    predictions = predict_model(other_texts)
    # print(predictions)
    logging.info('FIN main()')


if __name__ == "__main__":
    main()
