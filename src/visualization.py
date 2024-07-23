import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_average_family_counts(csv_file, n=5):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Melt the dataframe to long format
    df_melted = df.melt(id_vars=["Family"], var_name="Station", value_name="Count")

    # Extract year from station name
    df_melted["Year"] = df_melted["Station"].str[-4:]
    df_melted["Station"] = df_melted["Station"].str[8:-5]

    # Convert count to 10^7 cells/L
    df_melted["Count"] = df_melted["Count"] / 10000000

    # Calculate average and standard error for each family and year
    df_stats = (
        df_melted.groupby(["Family", "Year"])["Count"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    # Calculate overall average for sorting
    df_overall_avg = (
        df_stats.groupby("Family")["mean"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Get the top n families by overall average count
    top_families = df_overall_avg["Family"].head(n).tolist()

    # Filter data for top families
    df_plot = df_stats[df_stats["Family"].isin(top_families)]

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Create the grouped bar plot with error bars
    ax = sns.barplot(
        x="Family",
        y="mean",
        hue="Year",
        data=df_plot,
        order=top_families,
        capsize=0.1,
        errcolor="black",
        errwidth=1.5,
    )

    # Add error bars manually to ensure they're visible
    for i, year in enumerate(["2015", "2016"]):
        data = df_plot[df_plot["Year"] == year]
        data = (
            data.set_index("Family").loc[top_families].reset_index()
        )  # Ensure correct order
        x = range(len(top_families))
        plt.errorbar(
            [xi + (i - 0.5) * 0.4 for xi in x],
            data["mean"],
            yerr=data["sem"],
            fmt="none",
            capsize=5,
            ecolor="black",
            elinewidth=1.5,
        )

    # Customize the plot
    plt.title(f"Average Cell Counts for Top {n} Families", fontsize=16)
    plt.xlabel("Family", fontsize=12)
    plt.ylabel("Average Count (10⁷ cells / L)", fontsize=12)
    plt.legend(title="Year")

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_family_counts(csv_file, n=5):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Melt the dataframe to long format
    df_melted = df.melt(id_vars=["Family"], var_name="Station", value_name="Count")

    # Extract year from station name
    df_melted["Year"] = df_melted["Station"].str[-4:]
    df_melted["Station"] = df_melted["Station"].str[8:-5]

    # Convert count to 10^7 cells/L
    df_melted["Count"] = df_melted["Count"] / 10000000

    # Get the top n families by total count
    top_families = df_melted.groupby("Family")["Count"].sum().nlargest(n).index

    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(n))

    # Set up the plot
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(5 * grid_size, 3 * grid_size)
    )
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Plot each family
    for i, family in enumerate(top_families):
        family_data = df_melted[df_melted["Family"] == family]

        sns.barplot(x="Station", y="Count", hue="Year", data=family_data, ax=axes[i])

        axes[i].set_title(f"{family}")
        axes[i].set_ylabel("Count (10⁷ cells / L)")
        axes[i].legend(title="Year")

        # Rotate x-axis labels and show only station IDs
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)
        axes[i].set_xlabel("Station ID")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_environmental_variables(file_path):
    # Read the TSV file
    df = pd.read_csv(file_path, sep="\t")

    # Get the list of variables (excluding 'Station')
    variables = [col for col in df.columns if (col != "Station" and ">5" not in col)]

    # Group variables by their base name (without year)
    var_groups = {}
    for var in variables:
        base_name = var.split("_")[0]
        if base_name not in var_groups:
            var_groups[base_name] = []
        var_groups[base_name].append(var)

    # Set up the subplot grid
    n_vars = len(var_groups)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    # fig.suptitle('Comparison of Variables Across Stations (2015 vs 2016)', fontsize=16)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Plot each variable group
    for i, (base_name, vars) in enumerate(var_groups.items()):
        ax = axes[i]

        for var in vars:
            year = "2015" if "_15" in var else "2016"
            color = "C1" if year == "2015" else "C0"  # C1 is orange, C0 is blue
            ax.plot(
                df["Station"],
                df[var],
                marker="o",
                linestyle="-",
                label=f"{year}",
                color=color,
            )

        if base_name in ["chlorophyll_a", "DIC_uptake"]:
            ax.set_title(f"{base_name} <5um")
        else:
            ax.set_title(f"{base_name}")
        ax.set_xlabel("Station")
        ax.set_ylabel("Value")
        ax.legend()

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", rotation=45)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
