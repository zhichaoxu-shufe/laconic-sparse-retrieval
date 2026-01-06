import os
import argparse
from tqdm import tqdm

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "encodings.jsonl")

    input_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")
    )

    with open(output_file, "wb") as outfile:
        for filename in tqdm(input_files, desc="Concatenating"):
            with open(os.path.join(args.input_dir, filename), "rb") as infile:
                while True:
                    chunk = infile.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    outfile.write(chunk)