"""Generate text from a trained family checkpoint to make results visible."""
import sys

import torch

from experiments.train import Model, load_data


def gen(model, tok_enc, tok_dec, n=400, seed=0):
    torch.manual_seed(seed)
    model.eval()
    SEQ = 192
    with torch.no_grad():
        seq = torch.full((SEQ,), tok_enc["\n"], dtype=torch.long)
        out = []
        for _ in range(n):
            logits = model(seq.unsqueeze(0))
            p = torch.softmax(logits[0, -1] / 0.8, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            out.append(tok_dec[nxt])
            seq = torch.cat([seq[1:], torch.tensor([nxt], dtype=torch.long)])
        return "".join(out)


def main():
    fam = sys.argv[1]
    _, _, vocab = load_data()
    tok_enc = {c: i for i, c in enumerate(sorted(set(
        open("data/tinyshakespeare.txt", encoding="utf-8").read())))}
    tok_dec = {i: c for c, i in tok_enc.items()}
    print("=" * 20, fam, "=" * 20)
    model = Model(fam, vocab)
    model.load_state_dict(torch.load(f"results/{fam}_600.pt",
                                     map_location="cpu")["model"])
    print("TRAINED", fam)
    print(gen(model, tok_enc, tok_dec))


if __name__ == "__main__":
    main()