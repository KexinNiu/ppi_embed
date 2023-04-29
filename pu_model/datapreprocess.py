import torch
from pathlib import Path
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

def extract_esm(fasta_file, output_dir=Path('data/esm/'), model_location='esm2_t36_3B_UR50D',
                truncation_seq_length=1022, toks_per_batch=4096):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences",flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = False

    repr_layers = [36,]

    proteins = []
    data = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)",flush=True
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                result = {"label": label}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }
                proteins.append(label)
                data.append(result["mean_representations"][36])
    data = torch.stack(data).reshape(-1, 2560)
    return proteins, data

