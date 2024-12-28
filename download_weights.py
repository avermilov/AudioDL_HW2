import os

import gdown


def download():
    gdown.download(id="1K4iHBuLKO_rLBC0zlHJrpc_iqyEjpHMB")  # librispeech_lm_vocab.txt
    gdown.download(id="1T_dCqV1Mm46qqDwfyzxKiEn5hUgZzc8V")  # lower_4gram.arpa
    gdown.download(id="1jl9oYCFq2GrG184tv76_lbNsbJBzw8On")  # model_best.pth
    gdown.download(id="1CmVbDPo9h97lbfzOCIqagWHofTFZDDmN")  # tokenizer.json

    weights_dir = "data/weights/"
    os.makedirs(weights_dir, exist_ok=True)
    for filename in [
        "librispeech_lm_vocab.txt",
        "lower_4gram.arpa",
        "model_best.pth",
        "tokenizer.json",
    ]:
        os.rename(filename, os.path.join(weights_dir, filename))


if __name__ == "__main__":
    download()
