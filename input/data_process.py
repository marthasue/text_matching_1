import pandas as pd
from sklearn.utils import shuffle
#
df = pd.read_excel('all.xlsx', 'Sheet1')
for i in df.index:
    if pd.isna(df.loc[i, 'ent_name_host']) == True:
        host_name = ''
    else:
        host_name = df.loc[i, 'ent_name_host']
    if pd.isna(df.loc[i, 'ent_name_guest']) == True:
        guest_name = ''
    else:
        guest_name = df.loc[i, 'ent_name_guest']
    if pd.isna(df.loc[i, 'host_main_product']) == True:
        host_main_product = ''
    else:
        host_main_product = df.loc[i, 'host_main_product']
    if pd.isna(df.loc[i, 'host_ambi_product']) == True:
        host_ambi_product = ''
    else:
        host_ambi_product = df.loc[i, 'host_ambi_product']
    if pd.isna(df.loc[i, 'host_main_industry']) == True:
        host_main_industry = ''
    else:
        host_main_industry = df.loc[i, 'host_main_industry']
    if pd.isna(df.loc[i, 'host_ambi_industry']) == True:
        host_ambi_industry = ''
    else:
        host_ambi_industry = df.loc[i, 'host_ambi_industry']
    if pd.isna(df.loc[i, 'ent_name_guest']) == True:
        guest_name = ''
    else:
        guest_name = df.loc[i, 'ent_name_guest']
    if pd.isna(df.loc[i, 'guest_main_product']) == True:
        guest_main_product = ''
    else:
        guest_main_product = df.loc[i, 'guest_main_product']
    if pd.isna(df.loc[i, 'guest_ambi_product']) == True:
        guest_ambi_product = ''
    else:
        guest_ambi_product = df.loc[i, 'guest_ambi_product']
    if pd.isna(df.loc[i, 'guest_main_industry']) == True:
        guest_main_industry = ''
    else:
        guest_main_industry = df.loc[i, 'guest_main_industry']
    if pd.isna(df.loc[i, 'guest_ambi_industry']) == True:
        guest_ambi_industry = ''
    else:
        guest_ambi_industry = df.loc[i, 'guest_ambi_industry']

    df.loc[
        i, 'host'] = host_name + ':' + host_main_product + host_main_industry +'，想合作'+host_ambi_product + host_ambi_industry
    df.loc[
        i, 'guest'] = guest_name + ':' + guest_main_product + guest_main_industry +'，想合作'+ guest_ambi_product + guest_ambi_industry

df.to_excel('all_no.xlsx', 'Sheet1')
df1 = df.loc[:,['host','guest','label']]
df1.columns=['sentence1','sentence2','label']
df1 = shuffle(df1)
df_origin = df1
df_train = df_origin.iloc[:6862,:]
df_train = df_train.reset_index(drop=True)
df_test = df_origin.iloc[6862:,:]
df_test = df_test.reset_index(drop=True)
df_train.to_csv('train.csv',sep='\t')
df_test.to_csv('test.csv',sep='\t')
df_train.to_excel('TRAIN.xlsx','Sheet1')
df_test.to_excel('TEST.xlsx','Sheet1')