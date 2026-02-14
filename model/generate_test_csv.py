import os
from model.data_loader import load_drybean_from_uci

def generate_test_csv(sample_size=200):
    # Create /data folder at project root
    os.makedirs("data", exist_ok=True)

    # Load full dataset from UCI
    df = load_drybean_from_uci()

    # Sample small dataset
    sample_df = df.sample(sample_size, random_state=42)

    # Save to repo
    output_path = "data/test_data.csv"
    sample_df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")
    print("Rows:", sample_df.shape[0])
    print("Columns:", sample_df.shape[1])

if __name__ == "__main__":
    generate_test_csv(sample_size=500)
