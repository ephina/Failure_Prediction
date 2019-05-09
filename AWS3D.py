import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def main():

    df_dframe = pd.DataFrame()

    # read the file
    with open("D:/Official/aws/project/device_failure_work.csv", 'r', newline='', encoding='ISO-8859-1') as df_file:
        df_dframe = pd.read_csv(df_file)

    df_colnames = ['attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5', 'attribute6', 'attribute7',
                   'attribute8', 'attribute9']

    # slice the dependent variables and find their correlation
    df_attr = pd.DataFrame(df_dframe.loc[:,df_colnames])
    df_attr_corr = df_attr.corr(method='pearson')

    # remove one of the correlated columns
    df_dic={}
    for rows,cols in df_attr_corr.iterrows():
        for i in df_colnames:
            if(rows!=i and cols[i]>0.95 and i not in df_dic):
                df_dic[rows]=i
    df_colnames = set(df_colnames)-set(df_dic.values())
    print(df_colnames)
    #normalization of attribute values
    df_norm = pd.DataFrame(df_dframe.loc[:,df_colnames])
    df_norm = StandardScaler().fit_transform(df_norm)

    #PCA
    df_pca = PCA(n_components=None)
    df_prin_comps = df_pca.fit_transform(df_norm)
    no_prin_comp=0
    cum_var = 0
    for val in df_pca.explained_variance_ratio_:
        cum_var=cum_var+val
        if(cum_var<.95):
            no_prin_comp = no_prin_comp+1
    df_prin_sel = df_prin_comps[:,:no_prin_comp]

    #neural net

    # Preparing testing and training records in chronological order

    df_no_rows = len(df_prin_sel)

    df_train_no = int(.8 * df_no_rows)

    df_train_rec_x = df_prin_sel[:df_train_no,:]
    df_train_rec_y = df_dframe.loc[:df_train_no,'failure']

    df_test_rec_x = df_prin_sel[df_train_no+1:,:]
    df_test_rec_y = df_dframe.loc[df_train_no+1:, 'failure']

    df_neuro_class = MLPClassifier(solver='lbfgs',hidden_layer_sizes = (150, ), learning_rate_init=0.00001,random_state = 1,max_iter=10000, activation='relu')

    df_neuro_class.fit(df_test_rec_x,df_test_rec_y)

    df_pred_y = df_neuro_class.predict(df_test_rec_x)
    tn, fp, fn, tp = confusion_matrix(df_test_rec_y,df_pred_y).ravel()
    print("True Positives = " +str(tp))
    print("True Negatives = " +str(tn))
    print("False Positives = " +str(fp))
    print("False Negatives = " +str(fn))
main()