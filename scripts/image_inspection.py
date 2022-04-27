import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 200
import pandas as pd


merged_df = pd.read_csv(
    "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/lg12_dn4000cut_sf_qc_superdf/qc_lg12_global_df.csv"
)

# merged_df = lg12_merged_df
noisy_plateifu_list = []
i = 0
for plateifu in merged_df["plateifu"]:
    # print(plateifu)
    sdss_img_path = "/Users/mmckay/Desktop/research/MM_manga_maps_FITS_CSV/lg12_sdss_images/{}.png".format(
        plateifu
    )
    sdss_img_array = mpimg.imread(sdss_img_path)
    plt.imshow(sdss_img_array, origin="lower")
    plt.show(block=False)
    plt.pause(1)
    input("Press enter to continue...")

    i += 1
    bad_good_user_input = input(
        "[{}/{}]  Noisy Image?(y/n):".format(i, len(merged_df["plateifu"]))
    )
    print(bad_good_user_input.lower, [plateifu])

    if bad_good_user_input == "y":
        noisy_plateifu_list.append(plateifu)
        plt.close()
    elif bad_good_user_input == "n":
        plt.close()
        pass
    else:
        print("wrong input, try again")
        continue

print(noisy_plateifu_list)
mk_new_csv_user_response = input(
    "Make new csv with noisey data plateifu values removed?(y/n):"
)
sample_name_user_input = input("Which sample? (sf/qc):")
if mk_new_csv_user_response == "y":
    noisy_merged_df = merged_df[merged_df["plateifu"].isin(noisy_plateifu_list)]
    clean_merged_df = merged_df[~merged_df["plateifu"].isin(noisy_plateifu_list)]
    noisy_merged_df.to_csv(
        "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/lg12_dn4000cut_sf_qc_superdf/noisy_{}_lg12_global.csv".format(
            sample_name_user_input
        )
    )
    clean_merged_df.to_csv(
        "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/lg12_dn4000cut_sf_qc_superdf/clean_{}_lg12_global.csv".format(
            sample_name_user_input
        )
    )

else:
    print("Doing nothing")
