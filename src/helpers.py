import pandas as pd


def rename_exchanges(uniprok_model, model, mapping_file):
    # Load the mapping file
    mapping_df = pd.read_csv(mapping_file, sep="\t")

    # Create dictionaries for easy lookup
    id_mapping = dict(zip(mapping_df["original_id"], mapping_df["renamed_id"]))

    # Rename reactions and metabolites
    for original_id, renamed_id in id_mapping.items():
        if original_id in model.reactions:
            # Get the reaction object from the model
            rxn = model.reactions.get_by_id(original_id)

            # Get the corresponding reaction from the universal model
            if renamed_id in uniprok_model.reactions:
                uni_rxn = uniprok_model.reactions.get_by_id(renamed_id)

                # Rename the reaction ID and name
                rxn.id = uni_rxn.id
                rxn.name = uni_rxn.name

                # Rename the metabolites
                for met in rxn.metabolites:
                    if met.id in uniprok_model.metabolites:
                        uni_met = uniprok_model.metabolites.get_by_id(met.id)
                        met.id = uni_met.id
                        met.name = uni_met.name
                print(f"Renamed ID {original_id} to {renamed_id}.")
        else:
            print(f"Original ID {original_id} not found in the model.")


def assign_medium(model, medium_dict):
    for ex in model.exchanges:
        ex.lower_bound = 0
    for ex_id, flux in medium_dict.items():
        if ex_id in model.exchanges:
            ex = model.reactions.get_by_id(ex_id)
            ex.lower_bound = -flux
    return model


def write_taxa_file(
    sample_id: str,
    model_ids: list,
    abundances: list,
    model_paths: list,
    output_tsv: str = None,
):
    """
    Generate a pandas DataFrame and a TSV file with the specified structure.

    Parameters:
    - sample_id (str): The sample ID to use for all rows.
    - model_ids (list[str]): A list of model IDs.
    - abundances (list[float]): A list of abundances corresponding to each model ID.
    - model_paths (list[str]): A list of file paths for each model ID.
    - output_tsv (str): The output file path for the TSV file.
    """
    if not (len(model_ids) == len(abundances) == len(model_paths)):
        raise ValueError("All input lists must have the same length")

    data = {
        "sample_id": [sample_id] * len(model_ids),
        "id": model_ids,
        "abundance": abundances,
        "file": model_paths,
    }

    df = pd.DataFrame(data)

    if output_tsv:
        df.to_csv(output_tsv, sep="\t", index=False)

    return df
