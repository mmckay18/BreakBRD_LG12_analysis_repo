import pandas as pd
import glob

# Combine all dataframes into a super dataframe
lg12_csv_list = glob.glob(
    "/Users/mmckay/Desktop/research/FMR_MZR/lg12_MMfits/*_map.csv"
)
combo_csv_list = []
for lg12 in lg12_csv_list:
    lg12_df = pd.read_csv(lg12)
    print(lg12, lg12_df.shape)
    combo_csv_list.append(lg12_df)
