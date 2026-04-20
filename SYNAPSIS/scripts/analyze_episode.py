import pandas as pd
import numpy as np

def analyze_csv(path):
    df = pd.read_csv(path)
    print(f"Total Rows: {len(df)}")
    
    # 1. State Distribution
    print("\n--- State Distribution ---")
    state_counts = df['state_str'].value_counts()
    print(state_counts)
    
    # 2. Joint Angles Presence
    joint_cols = [f'joint_{i}' for i in range(7)]
    joints_present = all(col in df.columns for col in joint_cols)
    print(f"\n--- Joint Columns Present: {joints_present} ---")
    if joints_present:
        print("Sample Joint Values (Row 0):")
        print(df[joint_cols].iloc[0].to_dict())
        
    # 3. AWAIT_STABLE_PLACEMENT Transition Analysis
    print("\n--- AWAIT_STABLE_PLACEMENT Sequence ---")
    await_df = df[df['state_str'] == 'AWAIT_STABLE_PLACEMENT']
    if not await_df.empty:
        print(f"Total Steps in AWAIT_STABLE_PLACEMENT: {len(await_df)}")
        print("Steps indices:", await_df.index.tolist())
    else:
        print("AWAIT_STABLE_PLACEMENT phase filtered out entirely (or not reached).")

if __name__ == "__main__":
    csv_path = "output_data/debug/csvs/w0_e0_seed56.csv"
    try:
        analyze_csv(csv_path)
    except Exception as e:
        print(f"Error: {e}")
