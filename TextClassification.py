import pandas as pd
import numpy as np
# from sklearn.externals import joblib
import joblib
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import argparse
import os


# nltk.download()

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
def cmp_L1(x):
    val = []
    for i,k in enumerate(x):
        if k == lv_l1_std[i]:
            val.append('True')
        else:
            val.append('False')
    return val
def cmp_module(x):
    val = []
    for i,k in enumerate(x):
        if k == lv_module_std[i]:
            val.append('True')
        else:
            val.append('False')
    return val
def cmp_submodule(x):
    val = []
    for i,k in enumerate(x):
        if k == lv_submodule_std[i]:
            val.append('True')
        else:
            val.append('False')
    return val
def STD_L1(x):
    tt = lv_L1_code.loc[lv_L1_code['Level 1 Process'] == x]
    ttt = tt.iloc[0]
    ttt = ttt['Level 1 Process_code']
    return ttt
def STD_SubModule_change(x):
    tt =lv_SubProcess_code.loc[lv_SubProcess_code['Sub Process']==x]
    ttt = tt.iloc[0]
    ttt=ttt['Sub Process_code']
    return ttt
def STD_Module_change(x):
    tt =lv_Module_code.loc[lv_Module_code['Module']==x]
    ttt = tt.iloc[0]
    ttt=ttt['Module_code']
    return ttt
def STD_Txt_Code(l1,module,sub_module):
    L1 = list(map(STD_L1, l1))
    Module = list(map(STD_Module_change,module))
    SubModule = list(map(STD_SubModule_change,sub_module))
    return L1,Module,SubModule
def code(codefile):
    # lv_code = pd.read_csv('Code.csv')
    lv_code1 = pd.read_csv(codefile)
    lv_code1['Sub Process'].fillna('nan', inplace=True)
    lv_code1['Level 1 Process'].fillna('nan', inplace=True)
    lv_code1['Module'].fillna('nan', inplace=True)
    lv_SubProcess_code1 = pd.DataFrame(lv_code1[['Sub Process_code', 'Sub Process']])
    lv_Module_code1 = pd.DataFrame(lv_code1[['Module_code', 'Module']])
    lv_L1_code1 = pd.DataFrame(lv_code1[['Level 1 Process_code', 'Level 1 Process']])
    return lv_code1,lv_L1_code1,lv_Module_code1,lv_SubProcess_code1
def L1_change(x):
    tt =lv_L1_code.loc[lv_L1_code['Level 1 Process_code']==x]
    ttt = tt.iloc[0]
    ttt=ttt['Level 1 Process']
    return ttt
def SubModule_change(x):
    tt =lv_SubProcess_code.loc[lv_SubProcess_code['Sub Process_code']==x]
    ttt = tt.iloc[0]
    ttt=ttt['Sub Process']
    return ttt
def Module_change(x):
    tt =lv_Module_code.loc[lv_Module_code['Module_code']==x]
    ttt = tt.iloc[0]
    ttt=ttt['Module']
    return ttt
def Execute_Transform(args):
    filename,codefile1,output  = args.input,args.codefile,args.output
    global lv_code, lv_L1_code, lv_Module_code, lv_SubProcess_code
    global lv_l1_std, lv_module_std, lv_submodule_std
    lv_code, lv_L1_code, lv_Module_code, lv_SubProcess_code = code(codefile1)

    lv_csv = pd.read_csv(filename, na_values=["nan"])
    # print(lv_csv.head())
    lv_csv = lv_csv.loc[lv_csv['Process area'] == 'R2R']

    lv_csv['Process area'] = lv_csv['Process area'].str.lower()
    lv_csv['Object description'] = lv_csv['Object description'].str.lower()
    lv_csv['Long Description'] = lv_csv['Long Description'].str.lower()

    lv_csv['Process area'].fillna('nan', inplace=True)
    lv_csv['Object description'].fillna('nan', inplace=True)
    lv_csv['Long Description'].fillna('nan', inplace=True)
    lv_csv['Level-1-text'] = lv_csv.apply(
        lambda row: '{} {} {}'.format(row['Process area'], row['Object description'], row['Long Description']), axis=1)
    lv_csv['Level-1-text'] = lv_csv['Level-1-text'].str.lower()

    lv_csv['Level 1 Process'].fillna('nan', inplace=True)
    lv_csv['Level 1 Process'] = lv_csv['Level 1 Process'].str.replace(' ', '')
    lv_csv['Level 1 Process'] = lv_csv['Level 1 Process'].str.lower()

    lv_csv['Sub Process'].fillna('nan', inplace=True)
    lv_csv['Sub Process'] = lv_csv['Sub Process'].str.replace(' ', '')
    lv_csv['Sub Process'] = lv_csv['Sub Process'].str.lower()

    lv_csv['Module'].fillna('nan', inplace=True)
    lv_csv['Module'] = lv_csv['Module'].str.replace(' ', '')
    lv_csv['Module'] = lv_csv['Module'].str.lower()

    lv_l1_std, lv_module_std, lv_submodule_std = STD_Txt_Code(lv_csv['Level 1 Process'],lv_csv['Module'],lv_csv['Sub Process']  )
    # print(lv_csv.head())
    lv_csv_final = lv_csv[['Object description', 'Long Description', 'Process area']]
    lv_csv_final.reset_index(inplace=True)
    lv_csv_final = lv_csv_final.drop(labels=['index'], axis=1)
    # lv_csv_final.head()

    X_test = pd.DataFrame(lv_csv[['Level-1-text']])  # lv_final_text['Level-1-text']
    X_test.reset_index(inplace=True)
    # X_test = X_test['Text']
    X_test.drop(labels=['index'], axis=1, inplace=True)
    X_test = X_test['Level-1-text']

    loaded_model = joblib.load('L1.pkl')
    test = loaded_model.predict(X_test)
    L1_std_val = cmp_L1(test)
    test = list(map(L1_change, test))
    test1 = pd.DataFrame(test, columns=['Level 1 Process'])
    lv_csv_final = pd.concat([lv_csv_final, test1], axis=1)
    test = pd.Series(test)
    # X_test = test  +' '+ X_test
    X_test = pd.concat([test, X_test], axis=1, join='outer')
    X_test = X_test[0] + ' ' + X_test['Level-1-text']
    # X_test

    loaded_model = joblib.load('Module-S1.pkl')
    test = loaded_model.predict(X_test)
    module_std_value = cmp_module(test)
    test = list(map(Module_change, test))
    test1 = pd.DataFrame(test, columns=['Module'])
    lv_csv_final = pd.concat([lv_csv_final, test1], axis=1, join='outer')
    test = pd.Series(test)
    X_test = pd.concat([test, X_test], axis=1, join='outer')
    X_test[0].fillna('nan', inplace=True)
    X_test = X_test[0] + ' ' + X_test[1]

    loaded_model = joblib.load('SM-S1.pkl')
    test = loaded_model.predict(X_test)
    submodule_std_value = cmp_submodule(test)
    # np.mean(test == y_test)
    comb_std = pd.DataFrame(list(zip(L1_std_val,module_std_value,submodule_std_value)),columns=['L1_Correct','Module_correct','Sub-Module_Correct'])
    test = list(map(SubModule_change, test))
    test1 = pd.DataFrame(test, columns=['Sub Process'])
    lv_csv_final = pd.concat([lv_csv_final, test1,comb_std], axis=1, join='outer')
    # lv_csv_final.to_csv('Final_File.csv')output
    lv_csv_final.to_csv(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='csv.csv', help='Input file path')
    parser.add_argument('--codefile', type=str,
                        default='Code.csv', help='Code file path')

    parser.add_argument('--output', type=str,
                        default='Final_File.csv', help='Output file path')
    args = parser.parse_args()
    Execute_Transform(args)