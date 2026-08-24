import argparse
import os
import pandas as pd
from PIL import Image
from constants import OUTPUT_COLUMNS, TRANSCRIPTION_COLUMNS


def handle_image(img: Image) -> pd.DataFrame:
    """
    Transcribe the contents of each box depicted by the image and return a DataFrame with
    the results in the specified format.
    """
   
    # Placeholder: future transcription logic must return TRANSCRIPTION_COLUMNS.
    return pd.DataFrame(columns=TRANSCRIPTION_COLUMNS)

def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index records by Node Title and flag every repeated title without removing
    any rows.
    """
    df = df.set_index("Node Title", drop=False)
    duplicate_rows = df.index.duplicated(keep=False)
    flags_column = df.columns.get_loc("Flags")
    duplicate_flag = "Potential Dupelicate"

    for row_position, is_duplicate in enumerate(duplicate_rows):
        if not is_duplicate:
            continue

        current_flags = df.iloc[row_position, flags_column]
        if pd.isna(current_flags) or not current_flags:
            df.iloc[row_position, flags_column] = duplicate_flag
        elif duplicate_flag not in str(current_flags).split("; "):
            df.iloc[row_position, flags_column] = f"{current_flags}; {duplicate_flag}"

    return df

def main():

    # parse args
    parser = argparse.ArgumentParser(description="UMA Box Contents Transcription CLI")
    parser.add_argument("-i", "--path-to-inputs", default=os.getcwd(),
                        help="Path to the input, a directory of images.")
    parser.add_argument("-o", "--path-to-outputs", default=os.path.join(os.getcwd(), "outputs"), 
                        help="Output file path (should end in '.csv')")
    args = parser.parse_args()

    # prepare to transcribe images
    df = pd.DataFrame(columns=TRANSCRIPTION_COLUMNS)
    input_filenames = os.listdir(args.path_to_inputs)

    # transcribe images
    for filename in input_filenames:
        try:
            img = Image.open(os.path.join(args.path_to_inputs, filename))
            image_records = handle_image(img)
            missing_columns = set(TRANSCRIPTION_COLUMNS) - set(image_records.columns)
            if missing_columns:
                raise ValueError(
                    f"Transcription is missing required columns: {sorted(missing_columns)}"
                )
            df = pd.concat(
                [df, image_records[TRANSCRIPTION_COLUMNS]], ignore_index=True
            )
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    # check for duplicate node titles
    df["Flags"] = ""
    df = check_duplicate_rows(df)

    # save results
    df.to_csv(args.path_to_outputs, columns=OUTPUT_COLUMNS, index=False)

if __name__ == "__main__":
    main()
