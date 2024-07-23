import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_crossfeeding_network(
    avg_exchanges_file,
    ecoregion,
    year,
    flux_cutoff: float = 0.001,
    taxa_node_size: float = 3000,
    metabolite_node_size: float = 1500,
    edge_width_factor: float = 2,
    hidden_metabolites: list = [],
    figsize=(10, 10),
    output_file=None,
):
    # Read the averaged exchanges file
    df = pd.read_csv(avg_exchanges_file, sep="\t")

    # Filter for the specified ecoregion and year
    df = df[df["sample_id"] == f"{ecoregion}_{year}"]

    # Apply flux cutoff
    df = df[abs(df["flux"]) >= flux_cutoff]

    if hidden_metabolites:
        df = df[~df["metabolite"].isin(hidden_metabolites)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in df.iterrows():
        if row["direction"] == "export":
            G.add_edge(row["taxon"], row["metabolite"], weight=row["flux"])
        else:  # import
            G.add_edge(row["metabolite"], row["taxon"], weight=-row["flux"])

    # Separate taxa and metabolite nodes
    taxa_nodes = sorted(set(df["taxon"].unique()))
    metabolite_nodes = sorted(set(G.nodes()) - set(taxa_nodes))

    # Create a layout with taxa in an inner pentagon and metabolites in an outer circle
    pos = {}
    # Position taxa nodes in a pentagon
    for i, node in enumerate(taxa_nodes):
        angle = 2 * np.pi * i / len(taxa_nodes) - np.pi / 2  # Start from the top
        pos[node] = (np.cos(angle), np.sin(angle))

    # Position metabolite nodes in a circle
    for i, node in enumerate(metabolite_nodes):
        angle = 2 * np.pi * i / len(metabolite_nodes)
        pos[node] = (
            1.5 * np.cos(angle),
            1.5 * np.sin(angle),
        )  # 1.5 times the radius of taxa circle

    # Calculate edge widths based on flux, using a square root scale
    max_flux = max(abs(G[u][v]["weight"]) for u, v in G.edges())
    edge_widths = [
        edge_width_factor * np.sqrt(abs(G[u][v]["weight"]) / max_flux)
        for u, v in G.edges()
    ]

    # Set up the plot
    plt.figure(figsize=figsize)

    # Draw the network
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=taxa_nodes,
        node_size=taxa_node_size,
        node_color="lightcoral",
        alpha=0.8,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=metabolite_nodes,
        node_size=metabolite_node_size,
        node_color="lightblue",
        alpha=0.8,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color="gray",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        alpha=0.6,
    )

    # Add labels with slight offset to avoid overlap with nodes
    taxa_labels = {node: node for node in taxa_nodes}
    metabolite_labels = {node: node for node in metabolite_nodes}
    nx.draw_networkx_labels(G, pos, taxa_labels, font_size=10, font_weight="bold")
    nx.draw_networkx_labels(G, pos, metabolite_labels, font_size=8)

    plt.title(f"Metabolite Cross-feeding Network - {ecoregion} {year}")
    plt.axis("off")

    # Adjust plot limits to ensure all nodes are visible
    plt.xlim(-1.7, 1.7)
    plt.ylim(-1.7, 1.7)

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Network diagram saved to {output_file}")

    # Show the plot
    plt.show()


def create_flux_difference_network(
    flux_diff_file,
    ecoregion,
    flux_cutoff: float = 0.001,
    taxa_node_size: float = 3000,
    metabolite_node_size: float = 1500,
    metabolites: list = [],
    edge_width_factor: float = 2,
    edge_width_exp: float = 0.2,
    figsize=(10, 10),
    output_file=None,
):
    # Read the flux difference file
    df = pd.read_csv(flux_diff_file, sep="\t")

    # Filter for the specified ecoregion
    df = df[df["ecoregion"] == ecoregion]

    # Apply flux cutoff
    df = df[abs(df["flux_diff_2016-2015"]) >= flux_cutoff]

    if metabolites:
        df = df[df["metabolite"].isin(metabolites)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in df.iterrows():
        if row["direction"] == "export":
            G.add_edge(
                row["taxon"], row["metabolite"], weight=row["flux_diff_2016-2015"]
            )
        else:  # import
            G.add_edge(
                row["metabolite"], row["taxon"], weight=-row["flux_diff_2016-2015"]
            )

    # Separate taxa and metabolite nodes
    taxa_nodes = sorted(set(df["taxon"].unique()))
    metabolite_nodes = sorted(set(G.nodes()) - set(taxa_nodes))

    # Create a layout with taxa in an inner pentagon and metabolites in an outer circle
    pos = {}
    # Position taxa nodes in a pentagon
    for i, node in enumerate(taxa_nodes):
        angle = 2 * np.pi * i / len(taxa_nodes) - np.pi / 2  # Start from the top
        pos[node] = (np.cos(angle), np.sin(angle))

    # Position metabolite nodes in a circle
    for i, node in enumerate(metabolite_nodes):
        angle = 2 * np.pi * i / len(metabolite_nodes)
        pos[node] = (
            1.5 * np.cos(angle),
            1.5 * np.sin(angle),
        )  # 1.5 times the radius of taxa circle

    # Calculate edge widths based on absolute flux difference
    max_flux = max(abs(G[u][v]["weight"]) for u, v in G.edges())
    edge_widths = [
        edge_width_factor * (abs(G[u][v]["weight"]) / max_flux) ** edge_width_exp
        for u, v in G.edges()
    ]

    # Determine edge colors based on flux difference
    edge_colors = [
        "salmon" if G[u][v]["weight"] < 0 else "lightblue" for u, v in G.edges()
    ]

    # Set up the plot
    plt.figure(figsize=figsize)

    # Draw metabolite nodes and edges first
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=metabolite_nodes,
        node_size=metabolite_node_size,
        node_color="lightgray",
        alpha=0.8,
    )

    metabolite_edges = [
        (u, v) for (u, v) in G.edges() if u in metabolite_nodes or v in metabolite_nodes
    ]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=metabolite_edges,
        width=[
            edge_widths[i] for i, e in enumerate(G.edges()) if e in metabolite_edges
        ],
        edge_color=[
            edge_colors[i] for i, e in enumerate(G.edges()) if e in metabolite_edges
        ],
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        alpha=0.6,
        arrows=True,
        node_size=metabolite_node_size,
    )

    # Draw taxa nodes and edges second
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=taxa_nodes,
        node_size=taxa_node_size,
        node_color="lightgray",
        alpha=0.8,
    )

    taxa_edges = [(u, v) for (u, v) in G.edges() if u in taxa_nodes and v in taxa_nodes]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=taxa_edges,
        width=[edge_widths[i] for i, e in enumerate(G.edges()) if e in taxa_edges],
        edge_color=[edge_colors[i] for i, e in enumerate(G.edges()) if e in taxa_edges],
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        alpha=0.6,
        arrows=True,
        node_size=taxa_node_size,
    )

    # Add labels with slight offset to avoid overlap with nodes
    taxa_labels = {node: node for node in taxa_nodes}
    metabolite_labels = {node: node for node in metabolite_nodes}
    nx.draw_networkx_labels(G, pos, taxa_labels, font_size=10, font_weight="bold")
    nx.draw_networkx_labels(G, pos, metabolite_labels, font_size=8)

    plt.title(f"2015 (El Niño) vs. 2016 - {ecoregion}")
    plt.axis("off")

    # Adjust plot limits to ensure all nodes are visible
    plt.xlim(-1.7, 1.7)
    plt.ylim(-1.7, 1.7)

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Network diagram saved to {output_file}")

    # Show the plot
    plt.show()


def plot_flux_difference_heatmap(
    flux_diff_file,
    ecoregion,
    flux_cutoff: float = 0.001,
    metabolites: list = [],
    figsize=(12, 8),
    output_file=None,
):
    # Read the flux difference file
    df = pd.read_csv(flux_diff_file, sep="\t")

    # Filter for the specified ecoregion
    df = df[df["ecoregion"] == ecoregion]

    # Apply flux cutoff
    df = df[abs(df["flux_diff_2016-2015"]) >= flux_cutoff]

    if metabolites:
        df = df[df["metabolite"].isin(metabolites)]

    # Prepare data for heatmap, using sum to aggregate duplicate entries
    heatmap_data = df.pivot_table(
        index="taxon", columns="metabolite", values="flux_diff_2016-2015", aggfunc="sum"
    )

    # Set up the plot
    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Flux Difference (2016 - 2015)"},
    )

    plt.title(f"Flux Differences Heatmap - {ecoregion}")
    plt.xlabel("Metabolites")
    plt.ylabel("Taxa")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {output_file}")

    # Show the plot
    plt.show()


def create_crossfeeding_dataset(avg_exchanges_file, output_file):
    # Read the averaged exchanges file
    df = pd.read_csv(avg_exchanges_file, sep="\t")

    # Split the sample_id into ecoregion and year
    df[["ecoregion", "year"]] = df["sample_id"].str.split("_", expand=True)

    # Create separate dataframes for exports and imports
    exports = df[df["direction"] == "export"]
    imports = df[df["direction"] == "import"]

    # Initialize a list to store results
    results = []

    # Iterate through each ecoregion and year combination
    for (ecoregion, year), group in df.groupby(["ecoregion", "year"]):
        exports_group = exports[
            (exports["ecoregion"] == ecoregion) & (exports["year"] == year)
        ]
        imports_group = imports[
            (imports["ecoregion"] == ecoregion) & (imports["year"] == year)
        ]

        # Get unique taxa
        taxa = group["taxon"].unique()

        # Iterate through all pairs of taxa
        for taxon1, taxon2 in itertools.combinations(taxa, 2):
            # Find shared metabolites
            exported_by_1 = set(
                exports_group[exports_group["taxon"] == taxon1]["metabolite"]
            )
            imported_by_2 = set(
                imports_group[imports_group["taxon"] == taxon2]["metabolite"]
            )
            shared_1_to_2 = exported_by_1.intersection(imported_by_2)

            exported_by_2 = set(
                exports_group[exports_group["taxon"] == taxon2]["metabolite"]
            )
            imported_by_1 = set(
                imports_group[imports_group["taxon"] == taxon1]["metabolite"]
            )
            shared_2_to_1 = exported_by_2.intersection(imported_by_1)

            # Add results if there are shared metabolites
            if shared_1_to_2:
                results.append(
                    {
                        "ecoregion": ecoregion,
                        "year": year,
                        "exporter": taxon1,
                        "importer": taxon2,
                        "shared_metabolites": len(shared_1_to_2),
                        "direction": f"{taxon1} -> {taxon2}",
                    }
                )
            if shared_2_to_1:
                results.append(
                    {
                        "ecoregion": ecoregion,
                        "year": year,
                        "exporter": taxon2,
                        "importer": taxon1,
                        "shared_metabolites": len(shared_2_to_1),
                        "direction": f"{taxon2} -> {taxon1}",
                    }
                )

    # Create a dataframe from the results
    result_df = pd.DataFrame(results)

    # Sort by number of shared metabolites (descending)
    result_df = result_df.sort_values("shared_metabolites", ascending=False)

    # Save to TSV
    result_df.to_csv(output_file, sep="\t", index=False)

    print(f"Cross-feeding dataset saved to {output_file}")

    return result_df


def visualize_crossfeeding_network(
    crossfeeding_file,
    ecoregion,
    year,
    node_size=3000,
    default_node_color="lightblue",
    edge_scale=0.5,
    min_shared_metabolites=1,
    figsize=(10, 10),
    output_file=None,
    colors=None,
):
    # Read the cross-feeding interactions file
    df = pd.read_csv(crossfeeding_file, sep="\t")

    # Filter for the specified ecoregion and year
    df_filtered = df[(df["ecoregion"] == ecoregion) & (df["year"] == year)]

    # Filter by minimum number of shared metabolites
    df_filtered = df_filtered[
        df_filtered["shared_metabolites"] >= min_shared_metabolites
    ]

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in df_filtered.iterrows():
        G.add_edge(row["exporter"], row["importer"], weight=row["shared_metabolites"])

    # Set up the plot
    plt.figure(figsize=figsize)

    # Create a pentagon layout
    pos = {
        node: (1.3 * np.cos(2 * np.pi * i / 5), 1.3 * np.sin(2 * np.pi * i / 5))
        for i, node in enumerate(sorted(G.nodes()))
    }

    # Prepare node colors
    if colors is None:
        node_colors = [default_node_color] * len(G.nodes())
    else:
        node_colors = [colors.get(node, default_node_color) for node in G.nodes()]

    # Draw the network
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors, alpha=0.8
    )

    # Draw edges with varying width based on shared metabolites
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        width=[w * edge_scale for w in weights],
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        node_size=node_size,
    )  # This creates space between arrows and nodes

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title(f"Cross-feeding Network - {ecoregion} {year}")
    plt.axis("off")

    # Expand the plot area to ensure all elements are visible
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Network diagram saved to {output_file}")

    # Show the plot
    plt.show()


def create_crossfeeding_difference(
    crossfeeding_file, output_file, year1=2015, year2=2016
):
    # Read the cross-feeding interactions file
    df = pd.read_csv(crossfeeding_file, sep="\t")

    # Filter for the two years of interest
    df_filtered = df[(df["year"] == year1) | (df["year"] == year2)]

    # Pivot the dataframe to have years as columns
    df_pivot = df_filtered.pivot_table(
        values="shared_metabolites",
        index=["ecoregion", "exporter", "importer"],
        columns="year",
        aggfunc="first",
    ).reset_index()

    # Rename the year columns
    df_pivot.rename(
        columns={
            year1: f"shared_metabolites_{year1}",
            year2: f"shared_metabolites_{year2}",
        },
        inplace=True,
    )

    # Calculate the difference (year2 - year1)
    df_pivot[f"diff_{year2}-{year1}"] = (
        df_pivot[f"shared_metabolites_{year2}"]
        - df_pivot[f"shared_metabolites_{year1}"]
    )

    # Fill NaN values with 0 (for cases where an interaction exists in one year but not the other)
    df_pivot = df_pivot.fillna(0)

    # Calculate percentage change
    df_pivot["percent_change"] = np.where(
        df_pivot[f"shared_metabolites_{year1}"] != 0,
        (df_pivot[f"diff_{year2}-{year1}"] / df_pivot[f"shared_metabolites_{year1}"])
        * 100,
        np.inf,  # Use infinity for cases where year1 value is 0
    )

    # Sort by absolute difference
    df_pivot["abs_diff"] = abs(df_pivot[f"diff_{year2}-{year1}"])
    df_pivot = df_pivot.sort_values("abs_diff", ascending=False).drop(
        "abs_diff", axis=1
    )

    # Save to TSV
    df_pivot.to_csv(output_file, sep="\t", index=False)

    print(f"Cross-feeding difference data saved to {output_file}")

    return df_pivot


def visualize_crossfeeding_difference(
    difference_file,
    ecoregion,
    year1,
    year2,
    node_size=3000,
    default_node_color="lightgray",
    edge_scale=0.5,
    min_difference=1,
    output_file=None,
    figsize=(10, 10),
    colors=None,
):
    # Read the cross-feeding difference file
    df = pd.read_csv(difference_file, sep="\t")

    # Filter for the specified ecoregion
    df_filtered = df[df["ecoregion"] == ecoregion]

    # Filter by minimum absolute difference
    df_filtered = df_filtered[
        abs(df_filtered[f"diff_{year2}-{year1}"]) >= min_difference
    ]

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in df_filtered.iterrows():
        G.add_edge(
            row["exporter"], row["importer"], weight=row[f"diff_{year2}-{year1}"]
        )

    # Set up the plot
    plt.figure(figsize=figsize)

    # Create a pentagon layout with adjusted radius
    radius = 1.0  # Reduced from 1.3
    pos = {
        node: (radius * np.cos(2 * np.pi * i / 5), radius * np.sin(2 * np.pi * i / 5))
        for i, node in enumerate(sorted(G.nodes()))
    }

    # Prepare node colors
    if colors is None:
        node_colors = [default_node_color] * len(G.nodes())
    else:
        node_colors = [colors.get(node, default_node_color) for node in G.nodes()]

    # Draw the network nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors, alpha=0.8
    )

    # Draw edges with varying width based on absolute difference and color based on sign
    edges = G.edges()
    weights = [abs(G[u][v]["weight"]) for u, v in edges]
    colors = ["salmon" if G[u][v]["weight"] < 0 else "lightblue" for u, v in edges]

    nx.draw_networkx_edges(
        G,
        pos,
        width=[w * edge_scale for w in weights],
        edge_color=colors,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        node_size=node_size,
    )  # This creates space between arrows and nodes

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Adjust plot limits to ensure all nodes are visible
    node_radius = (
        np.sqrt(node_size / np.pi) / 100
    )  # Approximate node radius in axis units
    plt.xlim(-radius - node_radius, radius + node_radius)
    plt.ylim(-radius - node_radius, radius + node_radius)

    # Increase figure size slightly
    fig = plt.gcf()
    # fig.set_size_inches(16, 16)

    plt.title(f"2015 (El Niño) vs. 2016 - {ecoregion}")
    plt.axis("off")

    # Expand the plot area to ensure all elements are visible
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Network diagram saved to {output_file}")

    # Show the plot
    plt.show()


def plot_flux_summary_heatmaps(
    flux_diff_file, flux_cutoff: float = 0.001, figsize=(20, 10), output_file=None
):
    # Read the flux difference file
    df = pd.read_csv(flux_diff_file, sep="\t")

    # Apply flux cutoff
    df = df[abs(df["flux_diff_2016-2015"]) >= flux_cutoff]

    # Separate imports and exports
    imports = df[df["direction"] == "import"]
    exports = df[df["direction"] == "export"]

    # Create summary tables
    import_summary = (
        imports.groupby(["ecoregion", "taxon"])["flux_diff_2016-2015"].sum().unstack()
    )
    export_summary = (
        exports.groupby(["ecoregion", "taxon"])["flux_diff_2016-2015"].sum().unstack()
    )

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)  # Changed to 1 row, 2 columns

    # Plot import heatmap
    sns.heatmap(
        import_summary,
        cmap="coolwarm",
        center=0,
        ax=ax1,
        cbar_kws={"label": "Total Import Flux Difference (2016 - 2015)"},
    )
    ax1.set_title("Average Import Flux Difference")
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # Plot export heatmap
    sns.heatmap(
        export_summary,
        cmap="coolwarm",
        center=0,
        ax=ax2,
        cbar_kws={"label": "Total Export Flux Difference (2016 - 2015)"},
    )
    ax2.set_title("Average Export Flux Difference")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Summary heatmaps saved to {output_file}")

    # Show the plot
    plt.show()


def plot_average_taxa_degree(flux_file, output_file=None):
    # Read the flux file
    df = pd.read_csv(flux_file, sep="\t")

    # Split sample_id into ecoregion and year
    df[["ecoregion", "year"]] = df["sample_id"].str.split("_", expand=True)

    # Calculate total degree for each taxon, ecoregion, and year
    total_degree = (
        df.groupby(["ecoregion", "taxon", "year"]).size().unstack(level="year")
    )

    # Calculate mean and standard deviation across ecoregions
    mean_degree = total_degree.groupby("taxon").mean()
    std_degree = total_degree.groupby("taxon").std()

    # Sort taxa based on 2015 degree from largest to smallest
    mean_degree = mean_degree.sort_values("2015", ascending=False)
    std_degree = std_degree.loc[mean_degree.index]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting
    x = range(len(mean_degree))
    width = 0.35

    # Plot bars
    ax.bar(
        [i - width / 2 for i in x],
        mean_degree["2015"],
        width,
        label="2015",
        color="salmon",
        yerr=std_degree["2015"],
        capsize=5,
    )
    ax.bar(
        [i + width / 2 for i in x],
        mean_degree["2016"],
        width,
        label="2016",
        color="lightblue",
        yerr=std_degree["2016"],
        capsize=5,
    )

    # Customize the plot
    ax.set_ylabel("Average Exchanges Degree")
    # ax.set_title('Average Taxa Degree Across Ecoregions: 2015 vs 2016')
    ax.set_xticks(x)
    ax.set_xticklabels(mean_degree.index, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Average degree summary plot saved to {output_file}")

    # Show the plot
    plt.show()
