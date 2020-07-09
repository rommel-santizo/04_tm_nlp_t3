import logging

from src.models.predict_model import predict_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
# x = pd.DataFrame({'content': [1, 2, 3]})

other_texts = [
    ['goverment', 'public', 'computer'],
    ['survey', 'response', 'eps']]


def main():
    logging.info('INI main()')
    predictions = predict_model(other_texts)
    print(predictions)
    logging.info('FIN main()')


if __name__ == "__main__":
    main()

# [([(1, 0.40035966), (2, 0.04932471), (3, 0.19829187), (4, 0.3497834)], [(89, [4]), (1588, [3, 1]), (11669, [1])], [(89, [(4, 0.9998604)]), (1588, [(1, 0.114946656), (3, 0.88484746)]), (11669, [(1, 0.99490696)])]),
#  ([(1, 0.3076685), (2, 0.050944075), (3, 0.24259552), (4, 0.39654383)], [(1297, [1, 4, 3]), (2319, [3]), (13759, [4])], [(1297, [(1, 0.5650432), (3, 0.13717405), (4, 0.28919938)]), (2319, [(3, 0.9989777)]), (13759, [(4, 0.9742257)])])]
