import json
import numpy as np
import torch
import soundfile as sf

from pathlib import Path

from nemo.collections.tts.parts.utils.tts_dataset_utils import BetaBinomialInterpolator
from nemo.collections.tts.models import FastPitchModel


last_ckpt = "/home/andrey/NeMo_notebooks/studio_pete_exp/FastPitch/2023-04-13_20-41-25/checkpoints/FastPitch--val_loss=1.3946-epoch=7899-last.ckpt"


spec_model = FastPitchModel.load_from_checkpoint(last_ckpt)
spec_model.eval().cuda()


def load_wav(audio_file, target_sr=None):
    with sf.SoundFile(audio_file, "r") as f:
        samples = f.read(dtype="float32")
        sample_rate = f.samplerate
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(
                samples, orig_sr=sample_rate, target_sr=target_sr
            )
    return samples.transpose()


for j, fold in enumerate(
    ["9017_manifest_train_dur_5_mins_local", "9017_manifest_dev_ns_all_local"]
):
    # Get records from the training manifest
    manifest_path = f"./{fold}.json"
    records = []
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            records.append(json.loads(line))

    beta_binomial_interpolator = BetaBinomialInterpolator()
    spec_model.eval()

    device = spec_model.device

    save_dir = Path(f"./{fold}_mels")

    save_dir.mkdir(exist_ok=True, parents=True)

    # Generate a spectrograms (we need to use ground truth alignment for correct matching between audio and mels)
    max_len = 0
    for i, r in enumerate(records):
        audio = load_wav(r["audio_filepath"])
        audio = torch.from_numpy(audio).unsqueeze(0).to(device)
        audio_len = torch.tensor(
            audio.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)
        print(audio.shape)
        max_len = max(max_len, audio.shape[1])
        # Again, our finetuned FastPitch model doesn't use multiple speakers,
        # but we keep the code to support it here for reference
        if spec_model.fastpitch.speaker_emb is not None and "speaker" in r:
            speaker = torch.tensor([r["speaker"]]).to(device)
        else:
            speaker = None

        with torch.no_grad():
            if "normalized_text" in r:
                text = spec_model.parse(r["normalized_text"], normalize=False)
            else:
                text = spec_model.parse(r["text"])

            text_len = torch.tensor(
                text.shape[-1], dtype=torch.long, device=device
            ).unsqueeze(0)

            spect, spect_len = spec_model.preprocessor(
                input_signal=audio, length=audio_len
            )

            # Generate attention prior and spectrogram inputs for HiFi-GAN
            attn_prior = (
                torch.from_numpy(
                    beta_binomial_interpolator(spect_len.item(), text_len.item())
                )
                .unsqueeze(0)
                .to(text.device)
            )

            spectrogram = spec_model.forward(
                text=text,
                input_lens=text_len,
                spec=spect,
                mel_lens=spect_len,
                attn_prior=attn_prior,
                speaker=speaker,
            )[0]

            save_path = save_dir / f"mel_{i}.npy"
            np.save(save_path, spectrogram[0].to("cpu").numpy())
            r["mel_filepath"] = str(save_path)

    if j == 0:
        hifigan_manifest_path = "hifigan_train_ft.json"
    else:
        hifigan_manifest_path = "hifigan_val_ft.json"
    with open(hifigan_manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # Please do the same for the validation json. Code is omitted.

    print("max_len:", max_len)
