from neural_network import NeuralNetwork


def main():
    input_list = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    target_list = [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
    predict_list = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    nn = NeuralNetwork()

    for _ in range(10):
        for input_data in enumerate(input_list):
            nn.train(input_data[1], target_list[input_data[0]])

    for predict in predict_list:
        output = nn.predict(predict)
        print("{} XOR {} => {}".format(predict[0], predict[1], output.result()))


if __name__ == '__main__':
    main()
