import json
import pandas as pd
from typing import List, Tuple, Optional

import json
import pandas as pd
from typing import List, Tuple, Optional

def curate_svamp_and_save_json(raw_json_path: str, output_path: str):
    """
    Processes raw SVAMP data using the established logic and saves it to JSON.
    """
    # 1. Load the raw data
    raw = json.load(open(raw_json_path, "r"))
    df = pd.DataFrame(raw)
    
    # 2. Data Correction 
    df.at[966, "Type"] = "Common-Division"
    
    # 3. Get operator count 
    df['OperationCount'] = df['Equation'].str.count(r'[+\-*/]')

    # 4. Helper for text cleaning
    def _ensure_qmark(s):
        s = str(s).strip()
        return s if s.endswith("?") else (s + "?")

    # 5. Base Question Creation (PromptWithoutExample) 
    df["PromptWithoutExample"] = (
        df["Body"].fillna("").astype(str).str.strip()
        + " "
        + df["Question"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip().apply(_ensure_qmark)

    # 6. Creation of PromptWithoutCot 
    demo_body = df.groupby("Type")["Body"].shift(-1).fillna(df.groupby("Type")["Body"].transform("first"))
    demo_q    = df.groupby("Type")["Question"].shift(-1).fillna(df.groupby("Type")["Question"].transform("first"))
    demo_ans  = df.groupby("Type")["Answer"].shift(-1).fillna(df.groupby("Type")["Answer"].transform("first")).astype(str)

    demo_bq = (
        demo_body.fillna("").astype(str).str.strip()
        + " "
        + demo_q.fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip().apply(_ensure_qmark)

    df["PromptWithoutCot"] = (
        "Q: " + demo_bq
        + " A: The answer is " + demo_ans + "."
        + " Q: " + df["PromptWithoutExample"]
        + " A:"
    )

    # 7. Creation of PromptWithCot 
    chain_of_thought = (
        "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
        "Each can has 3 tennis balls. How many tennis balls does he have now? "
        "A: Roger started with 5 balls. Each can has 3 balls, so total balls from cans = 2 * 3 = 6. "
        "Then total = 5 + 6 = 11. The answer is 11."
    )

    df["PromptWithCot"] = (
        chain_of_thought
        + " Q: "
        + df["PromptWithoutExample"]
        + " A:"
    )

    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path}")

def load_curated_data(json_path: str) -> pd.DataFrame:
    """Utility to load the curated JSON back into a DataFrame."""
    return pd.read_json(json_path)


def create_and_save_balanced_subset(
    input_json_path: str, 
    output_json_path: str, 
    n_samples: int = 3, 
    random_state: int = 42
) -> None:
    """
    Reads the curated dataset, creates a balanced subset based on OperationCount and Type,
    and saves the resulting subset to a new JSON file.
    """
    # Load the curated dataset
    df = pd.read_json(input_json_path)
    
    # Apply the sampling logic
    sampled_df = (
        df[df["OperationCount"] != 0]
        .groupby(['OperationCount', 'Type'], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), n_samples), random_state=random_state))
        .reset_index(drop=True)
    )
    
    # Save the subset to the specified output path
    sampled_df.to_json(output_json_path, orient="records", indent=4)
    print(f"Balanced subset (n={n_samples}) saved successfully to: {output_json_path}")