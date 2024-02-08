import transformers


def getpath():
    '\\'.join(transformers.__file__.split('\\')[:-1]) + '/models/llama/convert_llama_weights_to_hf.py'


def main():
    print('\\'.join(transformers.__file__.split('\\')[:-1]) + '\\models\\llama\\convert_llama_weights_to_hf.py')


if __name__ == "__main__":
    main()
