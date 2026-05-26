from src.data_loader import curate_svamp_and_save_json, create_and_save_balanced_subset
import os

# Klasör yollarını tanımla
RAW_DATA_PATH = "data/raw/SVAMP.json"
PROCESSED_DATA_PATH = "data/processed/svamp_curated.json"
SUBSET_DATA_PATH = "data/processed/svamp_curated_subset.json"

# Processed klasörü yoksa oluştur
os.makedirs("data/processed", exist_ok=True)

if __name__ == "__main__":
    # 1. Adım: Ham veriyi CoT ve No-CoT formatına dönüştür
    print("Curating SVAMP dataset...")
    curate_svamp_and_save_json(
        raw_json_path=RAW_DATA_PATH, 
        output_path=PROCESSED_DATA_PATH
    )

    # 2. Adım: Deneyler için dengeli alt kümeyi oluştur
    print("Creating balanced subset for experiments...")
    create_and_save_balanced_subset(
        input_json_path=PROCESSED_DATA_PATH,
        output_json_path=SUBSET_DATA_PATH,
        n_samples=4,
        random_state=42
    )
    
    print("Data preparation complete!")