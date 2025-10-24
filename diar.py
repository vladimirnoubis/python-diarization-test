import os
import argparse
import subprocess
import time
import datetime
import uuid
import re
import json
import gc

import torch
import torchaudio
import pandas as pd
import numpy as np
import time

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from pyannote.audio import Pipeline


def speech_to_text(
    audio_file_wav: str,
    whisper_model: WhisperModel,
    diarization_pipeline: Pipeline,
    num_speakers: int = None,
    prompt: str = "",
    language: str = None,
):
    time_start = time.time()

    # 1. Transkripcija sa faster-whisper
    print("Početak transkripcije...")
    options = dict(
        language=language,
        beam_size=7,
        vad_filter=True,
        vad_parameters=VadOptions(
            max_speech_duration_s=30,
            min_speech_duration_ms=200,
            speech_pad_ms=100,
            threshold=0.1,
            neg_threshold=0.1,
        ),
        word_timestamps=True,
        initial_prompt=prompt,
        task="transcribe",
    )
    segments, transcript_info = whisper_model.transcribe(audio_file_wav, **options)

    # Konvertovanje generatora u listu i formatiranje reči
    segments = [
        {
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
        }
        for s in segments
    ]

    time_transcribing_end = time.time()
    print(
        f"Transkripcija završena za {time_transcribing_end - time_start:.2f}s. Pronađeno {len(segments)} segmenata."
    )

    # 2. Diarizacija sa pyannote.audio
    print("Početak diarizacije...")
    waveform, sample_rate = torchaudio.load(audio_file_wav)
    diarization = diarization_pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        # min_speakers=4, max_speakers=6
        num_speakers=5,
    )

    time_diarization_end = time.time()
    print(
        f"Diarizacija završena za {time_diarization_end - time_transcribing_end:.2f}s."
    )

    print("Početak spajanja rezultata...")

    diarize_segments = []
    for turn, speaker in diarization.speaker_diarization:
        diarize_segments.append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )

    if not diarize_segments:
        print("Upozorenje: Diarizacija nije pronašla nijednog govornika.")
        # Vrati samo transkript bez informacija o govornicima
        for seg in segments:
            seg["speaker"] = "UNKNOWN"
        return segments, 0, transcript_info.language

    diarize_df = pd.DataFrame(diarize_segments)
    unique_speakers = diarize_df["speaker"].unique()
    detected_num_speakers = len(unique_speakers)

    # 3. Spajanje rezultata transkripcije i diarizacije
    final_segments = []
    for segment in segments:
        # Pronađi koji govornik je najviše pričao tokom ovog segmenta
        dia_tmp = diarize_df[
            (diarize_df["start"] < segment["end"])
            & (diarize_df["end"] > segment["start"])
        ]

        if len(dia_tmp) > 0:
            # Izračunaj preklapanje za svakog govornika u segmentu
            intersections = np.minimum(dia_tmp["end"], segment["end"]) - np.maximum(
                dia_tmp["start"], segment["start"]
            )
            speaker = dia_tmp.loc[intersections.idxmax()]["speaker"]
        else:
            speaker = "UNKNOWN"

        new_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker": speaker,
        }
        final_segments.append(new_segment)

    # 4. Pametno grupisanje segmenata
    if final_segments:
        grouped_segments = []
        current_group = final_segments[0].copy()
        sentence_end_pattern = r"[.!?]+$"

        for segment in final_segments[1:]:
            # Uslovi za spajanje: isti govornik, mala pauza, trenutna rečenica se ne završava
            can_combine = (
                segment["speaker"] == current_group["speaker"]
                and (segment["start"] - current_group["end"]) < 1.0
                and not re.search(sentence_end_pattern, current_group["text"])
            )

            if can_combine:
                current_group["end"] = segment["end"]
                current_group["text"] += " " + segment["text"]
            else:
                grouped_segments.append(current_group)
                current_group = segment.copy()

        grouped_segments.append(current_group)
        final_segments = grouped_segments
        time_merging_end = time.time()
    print(f"Spajanje završeno za {time_merging_end - time_diarization_end:.2f}s.")

    return final_segments, detected_num_speakers, transcript_info.language


def main():
    print(f"PyTorch verzija: {torch.__version__}")
    print(f"CUDA dostupna: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch kompajliran za CUDA verziju: {torch.version.cuda}")
        print(f"Pronađena cuDNN verzija: {torch.backends.cudnn.version()}")
        print(f"Naziv GPU-a: {torch.cuda.get_device_name(0)}")
    parser = argparse.ArgumentParser(
        description="Transkripcija i diarizacija audio fajla."
    )
    parser.add_argument("audio_file", type=str, help="Putanja do ulaznog audio fajla.")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Oznaka jezika (npr. 'sr', 'en'). Automatska detekcija ako se ne navede.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Broj govornika. Automatska detekcija ako se ne navede.",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Početni prompt za Whisper model."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token za pyannote model.",
    )
    args = parser.parse_args()

    # Provera tokena
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token mora biti prosleđen preko --hf_token argumenta ili HF_TOKEN environment varijable."
        )

    # --- UČITAVANJE MODELA (Ekvivalent `setup()` funkciji) ---
    print("Učitavanje modela... Ovo može potrajati.")
    start = time.time()
    try:
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        ).to(torch.device("cuda"))
    except Exception as e:
        print(f"Greška prilikom učitavanja modela: {e}")
        return

    print("Modeli su uspešno učitani.")

    temp_wav_file = f"temp_{time.time_ns()}.wav"
    try:
        # --- PRIPREMA AUDIO FAJLA ---
        print(
            f"Konvertovanje audio fajla '{args.audio_file}' u format pogodan za obradu..."
        )
        original_audio_file = args.audio_file
        temp_128kbps_file = "temp_standardized.mp3"

        # Naredba za prvi korak
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                original_audio_file,
                "-b:a",
                "128k",  # Forsiraj audio bitrate na 128 kbps
                "-ar",
                "16000",  # Postavi sample rate na 16000 Hz
                "-ac",
                "1",  # Postavi mono kanal
                "-y",  # Prepiši fajl ako postoji
                temp_128kbps_file,
            ]
        )

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_128kbps_file,
                "-ar",
                "16000",  # Osiguraj da je sample rate 16000 Hz
                "-ac",
                "1",  # Osiguraj mono kanal
                "-c:a",
                "pcm_s16le",  # Formatiraj u standardni WAV
                temp_wav_file,
            ]
        )

        segments, num_speakers, language = speech_to_text(
            temp_wav_file,
            whisper_model,
            diarization_pipeline,
            args.num_speakers,
            args.prompt,
            args.language,
        )

        # --- PRIPREMA I ISPIS REZULTATA ---
        result = {
            "detected_language": language,
            "detected_num_speakers": num_speakers,
            "segments": segments,
        }

        # Ispis rezultata kao formatiran JSON
        print("\n--- REZULTAT ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except subprocess.CalledProcessError as e:
        print(
            "\nGreška sa ffmpeg komandom. Da li je ffmpeg instaliran i dostupan u PATH-u?"
        )
        print(f"ffmpeg stderr: {e.stderr.decode()}")
    except Exception as e:
        print(f"\nDošlo je do greške tokom obrade: {e}")
    finally:
        # --- ČIŠĆENJE ---
        print("\nČišćenje privremenih fajlova...")
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
        if os.path.exists(temp_128kbps_file):
            os.remove(temp_128kbps_file)
        # Oslobađanje memorije
        del whisper_model
        del diarization_pipeline
        gc.collect()
        torch.cuda.empty_cache()

        end = time.time()
    print(f"Gotovo. Izvrsavanje trajalo {end - start:.3f} sekundi")


if __name__ == "__main__":
    main()
