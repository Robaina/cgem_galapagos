from cobra.medium import minimal_medium
import os
from typing import Dict
from cobra import Model


def create_minimal_medium_files(
    gems: Dict[str, Model], output_dir: str
) -> Dict[str, Dict[str, float]]:
    """
    Create TSV files containing minimal medium definitions for all models in the given dictionary.

    This function calculates the minimal medium for each model, saves it as a separate TSV file
    named '{model_id}_minimal_medium.tsv' in the specified output directory, and returns a dictionary
    of minimal media.

    Args:
        gems (Dict[str, Model]): A dictionary where keys are model IDs and values are cobra Model objects.
        output_dir (str): The directory where TSV files will be saved.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are model IDs and values are dictionaries
        representing minimal media (exchange reaction IDs as keys and flux values as values).

    Raises:
        Exception: If there's an error in calculating minimal medium or writing to file.
    """
    os.makedirs(output_dir, exist_ok=True)
    minimal_media_dict = {}

    for model_id, model in gems.items():
        try:
            max_growth = model.slim_optimize()
            min_medium = minimal_medium(model, max_growth)

            df = min_medium.to_frame(name="Flux")
            df.index.name = "Reaction ID"

            filename = f"{model_id}_minimal_medium.tsv"
            filepath = os.path.join(output_dir, filename)

            df.to_csv(filepath, sep="\t")

            minimal_media_dict[model_id] = min_medium.to_dict()

            print(f"Created minimal medium file for {model_id}")
        except Exception as e:
            print(f"Error processing model {model_id}: {str(e)}")

    print("All minimal medium files have been created.")
    return minimal_media_dict
