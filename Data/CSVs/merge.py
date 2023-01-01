import os
import argparse
import pandas as pd

if __name__ == "__main__":
    # Parse input dir and destination file from arguments
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one.")
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    args = parser.parse_args()

    # Load CSVs using pandas and concatenate them
    df = pd.concat([pd.read_csv(f"{args.input}/{f}") for f in os.listdir(args.input)])

    more_than_10 = df["Number of images"] >= 10
    df = df[more_than_10]

    not_ny = df["Collection Name"] != "COVID-19-NY-SBU"
    df = df[not_ny]

    # Save the resulting CSV to the output file
    df.to_csv(args.output, index=False)

    print("Done merging!")
