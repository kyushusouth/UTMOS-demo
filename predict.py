import argparse
import pathlib

import polars as pl
import torch
import torchaudio
import tqdm
from torch.utils.data import DataLoader, Dataset

from score import Score


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", required=False, default=None, type=int)
    parser.add_argument(
        "--mode", required=True, choices=["predict_file", "predict_dir"], type=str
    )
    parser.add_argument(
        "--ckpt_path",
        required=False,
        default="epoch=3-step=7459.ckpt",
        type=pathlib.Path,
    )
    parser.add_argument("--inp_dir", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--inp_path", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--out_path", required=False, type=pathlib.Path)
    parser.add_argument("--num_workers", required=False, default=0, type=int)
    return parser.parse_args()


class Dataset(Dataset):
    def __init__(self, dir_path: pathlib.Path):
        self.wavlist = []

        # dir_path = pathlib.Path("/home/minami/lip2sp/results/base_hubert")
        # date_lst = [
        #     "20240621_134621",
        #     "20240621_155144",
        #     "20240621_202419",
        #     "20240622_003027",
        #     "20240622_103111",
        #     "20240623_001016",
        #     "20240622_161416",
        # ]
        # for date in date_lst:
        #     self.wavlist += (dir_path / date).glob("**/*.wav")

        self.wavlist += dir_path.glob("**/*.wav")

        _, self.sr = torchaudio.load(self.wavlist[0])

    def __len__(self):
        return len(self.wavlist)

    def __getitem__(self, idx):
        fname = self.wavlist[idx]
        wav, _ = torchaudio.load(fname)
        sample = fname.parents[0].name
        speaker = fname.parents[1].name
        date = fname.parents[2].name
        kind = fname.stem
        sample = {
            "wav": wav,
            "date": date,
            "speaker": speaker,
            "sample": sample,
            "kind": kind,
        }
        return sample

    def collate_fn(self, batch):
        max_len = max([x["wav"].shape[1] for x in batch])
        out = []
        date_lst = [t["date"] for t in batch]
        speaker_lst = [t["speaker"] for t in batch]
        sample_lst = [t["sample"] for t in batch]
        kind_lst = [t["kind"] for t in batch]
        # Performing repeat padding
        for t in batch:
            wav = t["wav"]
            amount_to_pad = max_len - wav.shape[1]
            padding_tensor = wav.repeat(1, 1 + amount_to_pad // wav.size(1))
            out.append(torch.cat((wav, padding_tensor[:, :amount_to_pad]), dim=1))
        return torch.stack(out, dim=0), date_lst, speaker_lst, sample_lst, kind_lst


def main():
    args = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "predict_file":
        assert (
            args.inp_path is not None
        ), "inp_path is required when mode is predict_file."
        assert args.inp_dir is None, "inp_dir should be None."
        assert args.inp_path.exists()
        assert args.inp_path.is_file()
        wav, sr = torchaudio.load(args.inp_path)
        scorer = Score(ckpt_path=args.ckpt_path, input_sample_rate=sr, device=device)
        score = scorer.score(wav.to(device))
        print(score[0])
        # with open(args.out_path, "w") as fw:
        #     fw.write(str(score[0]))
    else:
        dataset = Dataset(dir_path=args.inp_dir)
        loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            num_workers=args.num_workers,
        )
        sr = dataset.sr
        scorer = Score(
            ckpt_path="/home/minami/UTMOS-demo/epoch=3-step=7459.ckpt",
            input_sample_rate=sr,
            device=device,
        )

        results = []
        for batch in tqdm.tqdm(loader):
            wav, date_lst, speaker_lst, sample_lst, kind_lst = batch
            with torch.no_grad():
                scores = scorer.score(wav.to(device))
            for s, date, speaker, sample, kind in zip(
                scores, date_lst, speaker_lst, sample_lst, kind_lst
            ):
                results.append([s, date, speaker, sample, kind])
            # with open(args.out_path, "a") as fw:
            #     for s, date, speaker, sample, kind in zip(
            #         scores, date_lst, speaker_lst, sample_lst, kind_lst
            #     ):
            #         fw.write(f"{date},{speaker},{sample},{kind},{str(s)}\n")
        df = pl.DataFrame(
            data=results, schema=["score", "date", "speaker", "filename", "kind"]
        )
        df.write_csv(str(args.out_path))


if __name__ == "__main__":
    main()
