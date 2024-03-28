import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from base_utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, FactorAnalyzer, calculate_kmo
import plotly.express as px
plt.rcParams["font.sans-serif"] = ["SimHei"] #设置字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号


class analyzer:
    def __init__(self,type='local'):
        """
            参数:
                type (str): 数据加载模式。默认为本地模式
        """
        self.type = type
        self.data = self.load_data()

    def load_data(self):
        if self.type=='db':
            data = self.load_mysql()
            data = self.data_clean(data)
        elif self.type =='local':
            userinfo_df = pd.read_csv(r'../data/userinfo.csv')
            phoneinfo_df = pd.read_csv(r'../data/phoneinfo.csv')
            serviceuseageinfo_df = pd.read_csv(r'../data/serviceuseageinfo.csv')
            data = {
                'userinfo': userinfo_df,
                'phoneinfo': phoneinfo_df,
                'serviceuseageinfo': serviceuseageinfo_df
            }
        df = data['userinfo'].merge(data['phoneinfo'],on='user_id').merge(data['serviceuseageinfo'],on='user_id')
        df = df.drop(columns='user_id')
        # df.to_csv(r'E:\课程作业\毕设\基于机器学习的电信用户价值分析与预测\clean.csv',index = False)
        return df

    def load_mysql(self):
        conn = mysql_db()
        try:
            userinfo_df = pd.read_sql("select * from user_info", conn)
            phoneinfo_df = pd.read_sql("select * from phone_info", conn)
            serviceuseageinfo_df = pd.read_sql("select * from service_useage_info", conn)
            data_dic = {
                'userinfo': userinfo_df,
                'phoneinfo': phoneinfo_df,
                'serviceuseageinfo': serviceuseageinfo_df
            }
            return data_dic
        finally:
            conn.close()

    def data_explore(self):
        for k,v in self.data.items():
            print(f'{k}...........................')
            for col in v.columns.values:
                if col == 'user_id':
                    continue
                print(col)
                print(v[col].describe())
                print(v[col].value_counts())
                print()

    def data_clean(self,data_dic):
        data = data_dic
        def outliers_iqr(col_data,zero_boundary,multiple):
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiple * IQR
            upper_bound = Q3 + multiple * IQR
            normal_data = list()
            for i in col_data:
                outliter_tag = False
                if zero_boundary:
                    if i<=0:
                        outliter_tag = True
                else:
                    if i<0:
                        outliter_tag = True
                if i > upper_bound or outliter_tag:
                    continue
                else:
                    normal_data.append(i)
            replace_value = round(np.mean(normal_data))
            clean_coldata = list()
            for row in col_data:
                outliter_tag = False
                if zero_boundary:
                    if row<=0:
                        outliter_tag = True
                else:
                    if row<0:
                        outliter_tag = True
                if row > upper_bound or outliter_tag:
                    clean_coldata.append(replace_value)
                else:
                    clean_coldata.append(row)
            return clean_coldata

        for k,v in data.items():
            print(v.head())
            print(f'Cleaning...{k}...................')
            if k == 'userinfo':
                v.loc[v['marriage_counts']==-1,'marriage_counts'] = 0
                v.loc[v['region']==-1,'region'] = 18
                v.loc[v['activeusers_family'] > 10, 'activeusers_family'] = 10
                v.loc[v['adults_numbers_family'].isin([-1,0]),'adults_numbers_family'] = v.loc[v['adults_numbers_family'].isin([-1,0]),'activeusers_family']
                v.loc[v['adults_numbers_family']>6,'adults_numbers_family'] = 6
                v.loc[v['adults_numbers_family'] <= 0, 'adults_numbers_family'] = v['adults_numbers_family'].median()
                v.loc[v['expect_income'] == -1, 'expect_income'] = v['expect_income'].median()
                v.loc[v['has_creditcard'] == -1, 'has_creditcard'] = v['has_creditcard'].mode()[0]

            elif k == 'phoneinfo':
                v.loc[v['dualband_capability']==-1,'dualband_capability'] = v['dualband_capability'].mode()[0]
                v.loc[v['newphoneuser']==2,'newphoneuser'] = v['newphoneuser'].mode()[0]
                v.loc[v['phoneprice']<=199,'phoneprice'] = round(v['phoneprice'].mean())
                v.loc[v['phonenetwork']== -1,'phonenetwork'] = v['phonenetwork'].mode()[0]
                v.loc[v['phone_usedays']<=0,'phone_usedays'] = round(v['phone_usedays'].mean())

            elif k == 'serviceuseageinfo':
                v['user_values'] = outliers_iqr(v['user_values'],True,3)
                v['useminutes'] = outliers_iqr(v['useminutes'],False,3)
                v['over_useminutes'] = outliers_iqr(v['over_useminutes'],False,3)
                v['over_cost'] = outliers_iqr(v['over_cost'],False,3)
                v['voicecost'] = outliers_iqr(v['voicecost'],False,3)
                v.loc[v['overdata_cost']<0,'overdata_cost'] = v['overdata_cost'].mode()[0]
                v.loc[v['roaming_callcounts'] < 0, 'roaming_callcounts'] = v['roaming_callcounts'].mode()[0]
                playminute2negpercent = round(np.mean(v.loc[(v['useminutes_percentchange_before_threemonth']>-100)&(v['useminutes_percentchange_before_threemonth']<=0)]['useminutes_percentchange_before_threemonth']))
                playminute2pospercent = round(np.mean(v.loc[(v['useminutes_percentchange_before_threemonth']<=100)&(v['useminutes_percentchange_before_threemonth']>=0)]['useminutes_percentchange_before_threemonth']))
                v.loc[v['useminutes_percentchange_before_threemonth'] <= -100,'useminutes_percentchange_before_threemonth'] = playminute2negpercent
                v.loc[v['useminutes_percentchange_before_threemonth'] >= 100,'useminutes_percentchange_before_threemonth'] = playminute2pospercent
                playcost2negpercent = round(np.mean(v.loc[
                                                          (v['cost_percentchange_before_threemonth'] >= -100) & (
                                                                      v[
                                                                          'cost_percentchange_before_threemonth'] <= 0)]['cost_percentchange_before_threemonth']))
                playcost2pospercent = round(np.mean(v.loc[(v['cost_percentchange_before_threemonth'] <= 100) & (
                            v['cost_percentchange_before_threemonth'] >= 0)]['cost_percentchange_before_threemonth']))
                v.loc[v['cost_percentchange_before_threemonth'] <= -100, 'cost_percentchange_before_threemonth'] = playcost2negpercent
                v.loc[v['cost_percentchange_before_threemonth'] >= 100, 'cost_percentchange_before_threemonth'] = playcost2pospercent
                v['answercounts'] = outliers_iqr(v['answercounts'],False,3)
                v['inAndout_callcounts_PVC'] = outliers_iqr(v['inAndout_callcounts_PVC'],False,3)
                v['incomplete_minutes_PVC'] = outliers_iqr(v['incomplete_minutes_PVC'],False,3)
                v['callcounts_NPVC'] = outliers_iqr(v['callcounts_NPVC'],False,3)
                v['drop_callcounts'] = outliers_iqr(v['drop_callcounts'],False,3)

            # print(v.describe().loc['count'])
            v.to_csv(rf'../data/{k}.csv',index=False)
        return data

    def corr_analyzer(self):
        userinfo_cols = ['region', 'marriage_counts', 'adults_numbers_family', 'expect_income', 'has_creditcard',
                         'totalemployed_months', 'activeusers_family', 'credit_rating']
        phoneinfo_cols = ['dualband_capability', 'phoneprice', 'phonenetwork', 'newphoneuser', 'phone_usedays']
        serviceuseage_cols = [col for col in self.data.columns.values if col not in userinfo_cols and col not in phoneinfo_cols]
        serviceuseage_cols.append('totalemployed_months')
        serviceuseage_cols.append('phoneprice')
        serviceuseage_cols.append('phone_usedays')

        corr_matrix = self.data[serviceuseage_cols].corr()
        corr_matrix_df = pd.DataFrame(corr_matrix).set_index(corr_matrix.columns)
        feas_corr_dic = dict()
        for col in serviceuseage_cols:
            feas_corr_dic[col] = dict()
            feas_corr_dic[col]['middle'] = list()
            feas_corr_dic[col]['hight'] = list()
        mid_threshold =0.6
        hight_threshold =0.8
        for _col in corr_matrix_df.columns.values.tolist():
            for _index in corr_matrix_df.index.values.tolist():
                if _col == _index:
                    continue
                corr_value = corr_matrix_df.loc[_index,_col]
                if corr_value>=mid_threshold and corr_value<hight_threshold:
                    feas_corr_dic[_col]['middle'].append(_index)
                elif corr_value>=hight_threshold:
                    feas_corr_dic[_col]['hight'].append(_index)
        for k,v in feas_corr_dic.items():
            print(f'{k}===========')
            print('middle corr:',v['middle'])
            print('hight corr:',v['hight'])
            print()

        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix_df, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

        # fig = px.imshow(corr_matrix,
        #                 labels=dict(x="X轴标签", y="Y轴标签"),
        #                 x=corr_matrix_df.columns,
        #                 y=corr_matrix_df.columns,
        #                 color_continuous_scale='Viridis',
        #                 color_continuous_midpoint=0)
        # fig.update_layout(title='Correlation Heatmap')
        # fig.show()

    def factor_analyzer(self,action):
        userinfo_cols = ['region', 'marriage_counts', 'adults_numbers_family', 'expect_income', 'has_creditcard',
                         'totalemployed_months', 'activeusers_family', 'credit_rating']
        phoneinfo_cols = ['dualband_capability', 'phoneprice', 'phonenetwork', 'newphoneuser', 'phone_usedays']
        serviceuseage_cols = [col for col in self.data.columns.values if
                              col not in userinfo_cols and col not in phoneinfo_cols]
        serviceuseage_cols.remove('user_values')
        # factor_analyzer_cols = list()
        # for col in serviceuseage_cols:
        #     if 'billadjust' in col or 'avg' in col or 'lifecycle' in col:
        #         continue
        #     factor_analyzer_cols.append(col)
        # factor_analyzer_cols.remove('buzy_callcounts')
        # factor_analyzer_cols.remove('user_values')
        factor_data = self.data[serviceuseage_cols]
        if action=='check_data':
            kmo_all, kmo_model = calculate_kmo(factor_data)
            chi_square_value, p_value = calculate_bartlett_sphericity(factor_data)
            print(f'kmo:{kmo_model}')
            print(f'巴特利特球形检验结果|Chi-square value:{chi_square_value},P-value{p_value}')

        elif action=='get_factors_num':
            fa = FactorAnalyzer(n_factors=10,rotation='varimax')
            fa.fit(factor_data)
            eigenvalues = fa.get_eigenvalues()[0]
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, factor_data.shape[1] + 1), eigenvalues, marker='o')
            plt.title('Scree Plot')
            plt.xlabel('Factors')
            plt.ylabel('Eigenvalue')
            plt.grid(True)
            plt.show()

        elif action=='get_info':
            number = 5
            fa = FactorAnalyzer(n_factors=number,rotation='varimax')
            fa.fit(factor_data)
            factor_var = fa.get_factor_variance()
            print("每个因子的方差解释率：")
            for i in range(len(factor_var[0])):
                print(f"因子 {i + 1}: {factor_var[0][i]}")
            print("\n累计方差解释率：")
            for i in range(len(factor_var[1])):
                print(f"前 {i + 1} 个因子累计方差解释率：{factor_var[1][i]}")
            print("\n总方差：")
            print(f"总方差：{factor_var[2]}")
            dt2 = pd.DataFrame(fa.loadings_, index=serviceuseage_cols,columns=[ f'Factor{num}'for num in range(number)])
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(dt2, annot=True, cmap='BuPu')  # 这个cmap是指颜色区间，BuPu是一种颜色的范围，大致是由灰变紫
            plt.show()

        elif action=='get_data':
            number = 5
            fa = FactorAnalyzer(n_factors=number,rotation='varimax')
            fa.fit(factor_data)
            factor_scores = fa.transform(factor_data)
            factor_df = pd.DataFrame(factor_scores, columns=[f'Factor{num + 1}' for num in range(number)])
            concatenated_df = pd.concat([self.data, factor_df], axis=1)
            return concatenated_df


    def dim_analysis(self):
        def cut_level(data,x,interval):
            label_bins = [x for x in range(0, max(data[x]) + interval, interval)]
            label_levels = pd.cut(data[x], bins=label_bins, labels=False)
            label_levels = label_levels +1
            return label_levels

        def get_value_level(value):
            if value < one_level:
                value_level = 1
            elif value >= one_level and value < two_level:
                value_level = 2
            elif value >= two_level:
                value_level = 3
            return value_level

        dim_data = self.factor_analyzer('get_data')
        one_level = dim_data['user_values'].quantile(0.3)
        two_level = dim_data['user_values'].quantile(0.8)
        dim_data['value_level'] = dim_data['user_values'].apply(lambda x: get_value_level(x))
        dim_data['phone_usedays_level'] = cut_level(dim_data,'phone_usedays',365)
        dim_data['phoneprice_level'] = cut_level(dim_data,'phoneprice',499)
        dim_data['employed_level'] = cut_level(dim_data,'totalemployed_months',12)
        dim_data.loc[dim_data['employed_level']==6,'employed_level'] = 5
        for fa in [f'Factor{x+1}' for x in range(0,5)]:
            print(fa)
            a = dim_data.loc[(dim_data['employed_level']==3)]
            b = dim_data.loc[(dim_data['employed_level']==4)]
            # a = dim_data.loc[(dim_data['phone_usedays_level']==4)&(dim_data['value_level']==2)]
            # b = dim_data.loc[(dim_data['phone_usedays_level']==5)&(dim_data['value_level']==2)]
            print(np.mean(a[fa]))
            print(np.mean(b[fa]))
            print()
        f
        # self.show_bar(dim_data,'phone_usedays_level','Factor1','手机使用年数','增长系数')

        x_labels = 'employed_level'
        self.show_percent(dim_data,x=x_labels,hue='value_level')
        f
        self.show_scatter(dim_data,x='adults_numbers_family',y='employed_level',hue='value_level')

        # sns.scatterplot(data = dim_data,x='phone_usedays',y='phoneprice',hue='value_level')
        # plt.xlabel('手机使用天数')
        # plt.ylabel('手机价格')
        # plt.legend(title='用户价值', loc='upper left')
        # plt.show()

    def kmeans_cluster(self):
        clu_data = self.factor_analyzer('get_data')
        factors_col = [f'Factor{x+1}' for x in range(0,5)]
        factors_col.append('user_values')
        cluster_data = clu_data[factors_col]
        # cluster_data = StandardScaler().fit_transform(cluster_data)
        kmeans = KMeans(n_clusters=3,random_state=42)  # 指定聚类数目
        kmeans.fit(cluster_data)
        score = calinski_harabasz_score(cluster_data, kmeans.labels_)
        print("CH score:", score)
        values_label =  kmeans.labels_ + 1
        level_dic = {
            1:2,
            2:3,
            3:1,
        }
        user_level = [level_dic[x] for x in values_label]
        f
        self.data['kmeans_label'] = kmeans.labels_ + 1
        print(self.data['kmeans_label'].value_counts())

        a = self.data.groupby('kmeans_label')['user_values'].apply(lambda x:np.mean(x)).reset_index()
        for center in kmeans.cluster_centers_:
            print([round(x,4) for x in center])


    def show_bar(self,data,x,y,x_labs,y_labs):
        show_df = data.groupby(x)[y].apply(lambda x: round(x.mean(), 4)).reset_index()
        ax = sns.barplot(data=show_df, x=x, y=y)
        for index, row in show_df.iterrows():
            ax.text(row[x]-1, row[y], row[y], color='black', ha='center')
        plt.xlabel(x_labs)
        plt.ylabel(y_labs)
        plt.show()

    def show_percent(self,data, x, hue):
        df = data
        da = df.pivot_table(values='newphoneuser', aggfunc='count', index=x, columns=hue, fill_value=0,
                            margins=True)
        print(da)
        if isinstance(x, list):
            index_tag = [i for i in da.index.values if i[0] != 'All']
            col_tag = [i for i in da.columns.values if i != 'All']
        else:
            index_tag = [str(i) for i in da.index.values if i != 'All']
            col_tag = [i for i in da.columns.values if i != 'All']
        index_list = list()
        col_list = list()
        value = list()
        for i in index_tag:
            for j in range(0, len(col_tag)):
                if isinstance(x, list):
                    i = str(i)
                else:
                    i = int(i)
                index_list.append(i)
        for i in range(0, len(index_tag)):
            col_list.extend(col_tag)

        for i in index_tag:
            if isinstance(x,list):
                i = i
            else:
                i = int(i)
            for j in col_tag:
                percent = round(da.loc[i, j] * 100 / da.loc[i, 'All'], 2)
                value.append(percent)

        if isinstance(x, list):
            x_label = ','.join(x)
        else:
            x_label = x
        show_data = {
            x_label: index_list,
            hue: col_list,
            'percent': value,
        }
        show_df = pd.DataFrame(show_data)
        ax = sns.lineplot(data=show_df, x=x_label, y='percent', hue=hue, estimator=None)
        for index, row in show_df.iterrows():
            ax.text(row[x_label], row['percent'], row['percent'], color='black', ha='center')
        # plt.xlabel('用户在职年限(1年)')
        plt.xlabel('用户在职年数')
        plt.ylabel('用户价值占比(%)')
        plt.legend(title='用户价值', loc='upper left')
        plt.show()

    def show_scatter(self,data,hue,x=None,y=None,z=None):
        if z is None:
            group_list = [x, y]
            three_D = False
        else:
            group_list = [x,y,z]
            three_D = True
        grouped = data.groupby(group_list)[hue].value_counts(normalize=True).unstack().reset_index()
        hue_values = data[hue].unique()
        rename_dic ={}
        for hue_v in hue_values:
            rename_dic[hue_v] = f'{hue}_{hue_v}'
        grouped = grouped.rename(columns = rename_dic)
        grouped[list(rename_dic.values())[-1]] = grouped[list(rename_dic.values())[-1]]*100
        if three_D:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs=grouped[x],ys=grouped[y],zs=grouped[z],alpha=0.4, edgecolors='w')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
        else:
            sns.scatterplot(data=grouped,x=x,y=y,size=list(rename_dic.values())[-1],alpha=0.4, edgecolors='w')
        plt.xlabel('家庭成人数量')
        plt.ylabel('用户在职年限')
        plt.legend(title='用户价值占比', loc='upper left')
        plt.show()




if __name__ == '__main__':
    type = 'local' #db local
    tele_analyzer = analyzer(type)
    """
        相关性分析
    """
    # tele_analyzer.corr_analyzer()

    """
        因子分析   
        check_data：检查数据是否适合因子分析
        get_factors_num 获取最佳因子个数
        get_info 获取因子分解结果 
        get_data 生成因子分数
    """

    # tele_analyzer.kmeans_cluster()

    # tele_analyzer.factor_analyzer(action='get_info')

    """
        多维度分析
    """
    tele_analyzer.dim_analysis()

    # tele_analyzer.data_explore()