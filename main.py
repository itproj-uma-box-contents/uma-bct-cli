import argparse
import os
import pandas as pd
from PIL import Image
from constants import BOX_CONTENTS_COLUMNS


def handle_image(img: Image) -> pd.DataFrame:
    """
    Transcribe the contents of each box depicted by the image and return a DataFrame with
    the results in the specified format.
    """
   
    # Placeholder: future transcription logic must return BOX_CONTENTS_COLUMNS.
    return pd.DataFrame(columns=BOX_CONTENTS_COLUMNS)

def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are likely to represent the same box.
    """

    # placeholder
    return df.drop_duplicates()

def main():

    # parse args
    parser = argparse.ArgumentParser(description="UMA Box Contents Transcription CLI")
    parser.add_argument("-i", "--path-to-inputs", default=os.getcwd(),
                        help="Path to the input, a directory of images.")
    parser.add_argument("-o", "--path-to-outputs", default=os.path.join(os.getcwd(), "outputs"), 
                        help="Output file path (should end in '.csv')")
    args = parser.parse_args()

    # prepare to transcribe images
    df = pd.DataFrame(columns=BOX_CONTENTS_COLUMNS)
    input_filenames = os.listdir(args.path_to_inputs)

    # transcribe images
    for filename in input_filenames:
        try:
            img = Image.open(os.path.join(args.path_to_inputs, filename))
            image_records = handle_image(img)
            missing_columns = set(BOX_CONTENTS_COLUMNS) - set(image_records.columns)
            if missing_columns:
                raise ValueError(
                    f"Transcription is missing required columns: {sorted(missing_columns)}"
                )
            df = pd.concat(
                [df, image_records[BOX_CONTENTS_COLUMNS]], ignore_index=True
            )
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    # perform deduplication
    df = deduplicate_rows(df)

    # save results
    df.to_csv(args.path_to_outputs, index=False)

if __name__ == "__main__":
    main()
