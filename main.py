import argparse
import os
import pandas as pd
from PIL import Image

def handle_image(img: Image) -> pd.DataFrame:
    """
    Transcribe the contents of each box depicted by the image and return a DataFrame with
    the results in the specified format.
    """
   
    # placeholder
    return pd.DataFrame()

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
    parser.add_argument("-o", "--path-to-output", default="output.csv", 
                        help="Output file path (should end in '.csv')")
    args = parser.parse_args()

    # prepare to transcribe images
    df = pd.DataFrame([], columns=[
        # TODO: add column names here
    ])
    input_filenames = os.listdir(args.path_to_input)

    # transcribe images
    for filename in input_filenames:
        try:
            img = Image.open(os.path.join(args.path_to_input, filename))
            df = pd.concat([df, handle_image(img)], ignore_index=True)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    # perform deduplication
    df = deduplicate_rows(df)

    # save results
    df.to_csv(args.path_to_output, index=False)

if __name__ == "__main__":
    main()
