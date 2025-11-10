from argparse import ArgumentParser



def arguments():
    parser = ArgumentParser()
    parser.add_argument('server_model', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--hf-name', type=str)
    parser.add_argument('--hf-split', type=str)
    
    argv, _ = parser.parse_known_args()
    return argv


def main():
    args = arguments()

    # load_dataset()

    # load_model()


if __name__ == "__main__":
    main()
