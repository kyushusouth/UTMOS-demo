from score import Score
from model import load_ssl_model


def main():
    scorer = Score(
        ckpt_path="/home/minami/UTMOS-demo/epoch=3-step=7459.ckpt",
        input_sample_rate=16000,
        device="cpu",
    )
    ckpt_path = "/home/minami/UTMOS-demo/epoch=3-step=7459.ckpt"

    model = load_ssl_model(cp_path="wav2vec_small.pt")
    breakpoint()


if __name__ == "__main__":
    main()
