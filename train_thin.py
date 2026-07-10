import sys
sys.path.insert(0, ".")
import pandas as pd
from src.config import DATA_PROCESSED
from src.models.thin_file import train_thin_file

print("Loading processed features...")
df = pd.read_parquet(DATA_PROCESSED / "msme_features.parquet")
print(f"  {len(df)} rows loaded")
result = train_thin_file(df)
print(f"Done. AUC: {result['auc']:.4f}  Coverage: {result['coverage']:.3f}")
