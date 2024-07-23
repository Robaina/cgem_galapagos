import pandas as pd
import numpy as np
import argparse
import os


def main(table2_path, table3_path, output_path):
    # Read the CSV files
    table2 = pd.read_csv(table2_path)
    table3 = pd.read_csv(table3_path)

    # Create a dictionary to map sample IDs to station and year
    sample_map = dict(zip(table2["sample"], zip(table2["station"], table2["Year"])))

    # Get the abundance columns
    abundance_cols = [col for col in table3.columns if col.endswith("_abnd_perL")]

    # Create a list to store the processed data
    processed_data = []

    # Process each abundance column
    for col in abundance_cols:
        sample_id = col.split("_")[0]
        if sample_id in sample_map:
            station, year = sample_map[sample_id]
            for family, abundance in zip(table3["corrFamily"], table3[col]):
                processed_data.append(
                    {
                        "Family": family,
                        "Station": station,
                        "Year": year,
                        "Abundance": abundance,
                    }
                )

    # Convert the processed data to a DataFrame
    df = pd.DataFrame(processed_data)

    # Group by Family, Station, and Year, and calculate the mean abundance
    result = (
        df.groupby(["Family", "Station", "Year"])["Abundance"]
        .mean()
        .unstack(level=[1, 2])
    )

    # Rename columns to the desired format without decimals
    result.columns = [
        f"station_{int(station)}_{int(year)}" for station, year in result.columns
    ]

    # Separate 2015 and 2016 columns
    cols_2015 = [col for col in result.columns if col.endswith("_2015")]
    cols_2016 = [col for col in result.columns if col.endswith("_2016")]

    # Sort columns within each year
    cols_2015.sort(key=lambda x: int(x.split("_")[1]))
    cols_2016.sort(key=lambda x: int(x.split("_")[1]))

    # Combine sorted 2015 and 2016 columns
    new_cols = cols_2015 + cols_2016

    # Reindex the DataFrame with the new column order
    result = result.reindex(columns=new_cols)

    # Replace NaN with 0
    result = result.fillna(0)

    # Calculate total abundance for each family
    result["Total_Abundance"] = result.sum(axis=1)

    # Sort rows by total abundance in descending order
    result = result.sort_values("Total_Abundance", ascending=False)

    # Remove the Total_Abundance column
    result = result.drop("Total_Abundance", axis=1)

    # Save the result to a CSV file
    result.to_csv(output_path)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process microbial abundance data")
    parser.add_argument(
        "table2_path", help="Path to the CSV file containing sample metadata (table2)"
    )
    parser.add_argument(
        "table3_path", help="Path to the CSV file containing abundance data (table3)"
    )
    parser.add_argument("output_path", help="Path to save the output CSV file")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.table2_path):
        raise FileNotFoundError(f"Input file not found: {args.table2_path}")
    if not os.path.exists(args.table3_path):
        raise FileNotFoundError(f"Input file not found: {args.table3_path}")

    # Check if output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(args.table2_path, args.table3_path, args.output_path)
