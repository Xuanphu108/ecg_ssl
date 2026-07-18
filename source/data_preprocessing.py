from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path

target_fs = 100
data_root=Path("./datasets/ecg_data/")
target_root=Path("./datasets/ecg_data_chagas_processed")


data_folder_ribeiro_test = data_root/"ribeiro2020_test"
target_folder_ribeiro_test = target_root/("ribeiro_fs" + str(target_fs))

df_ribeiro_test, lbl_itos_ribeiro_test, mean_ribeiro_test, std_ribeiro_test = prepare_data_ribeiro_test(
    data_folder_ribeiro_test, 
    target_fs = target_fs, 
    channels = 12, 
    channel_stoi = channel_stoi_default, 
    target_folder = target_folder_ribeiro_test,
)

# reformat everything as memmap for efficiency
reformat_as_memmap(
    df_ribeiro_test, 
    target_folder_ribeiro_test/("memmap.npy"), 
    data_folder = target_folder_ribeiro_test, 
    delete_npys = True,
)


data_folder_zheng = data_root/"zheng2020/"
target_folder_zheng = target_root/("zheng_fs" + str(target_fs))

df_zheng, lbl_itos_zheng, mean_zheng, std_zheng = prepare_data_zheng(
    data_folder_zheng, 
    denoised = False, 
    target_fs = target_fs, 
    channels = 12, 
    channel_stoi = channel_stoi_default, 
    target_folder = target_folder_zheng,
)

# reformat everything as memmap for efficiency
reformat_as_memmap(
    df_zheng, 
    target_folder_zheng/("memmap.npy"), 
    data_folder = target_folder_zheng, 
    delete_npys = True,
)


data_folder_ptb_xl = data_root/"ptb_xl/"
target_folder_ptb_xl = target_root/("ptb_xl_fs" + str(target_fs))

df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(
    data_folder_ptb_xl, 
    min_cnt = 0, 
    target_fs = target_fs, 
    channels = 12, 
    channel_stoi = channel_stoi_default, 
    target_folder = target_folder_ptb_xl,
)

# reformat everything as memmap for efficiency
reformat_as_memmap(
    df_ptb_xl, 
    target_folder_ptb_xl/("memmap.npy"), 
    data_folder = target_folder_ptb_xl,
    delete_npys = True,
)



data_folder_cinc = data_root/"cinc2020/"
target_folder_cinc = target_root/("cinc_fs" + str(target_fs))

df_cinc, lbl_itos_cinc, mean_cinc, std_cinc = prepare_data_cinc(
    data_folder_cinc, 
    target_fs = target_fs, 
    channels = 12, 
    channel_stoi = channel_stoi_default, 
    target_folder = target_folder_cinc,
)

# reformat everything as memmap for efficiency
reformat_as_memmap(
    df_cinc, 
    target_folder_cinc/("memmap.npy"),
    data_folder = target_folder_cinc,
    delete_npys = True,
)

data_folder_code_15 = data_root/"code_15"
target_folder_code_15 = target_root/("code_15_fs" + str(target_fs))

df_code_15, lbl_itos_code_15, mean_code_15, std_code_15 = prepare_data_code_15_chagas(
    data_folder_code_15, 
    target_fs = target_fs, 
    channels = 12, 
    channel_stoi = channel_stoi_default, 
    target_folder = target_folder_code_15,
)

# reformat everything as memmap for efficiency
reformat_as_memmap(
    df_code_15, 
    target_folder_code_15/("memmap.npy"), 
    data_folder = target_folder_code_15, 
    delete_npys = True,
)