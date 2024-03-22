import datetime
import numpy as np
import optuna
import xgboost
import lightgbm
from sklearn.ensemble import RandomForestRegressor, StackingRegressor,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Lasso
from lazypredict.Supervised import LazyRegressor
import catboost
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import RFECV ,SelectFromModel,SequentialFeatureSelector
from sklearn.metrics import make_scorer,r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate,learning_curve
import os
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'

def show_percent(data,x,hue):
    df =data
    index_col = 'marriage_counts'
    column_col = 'value_level'
    da = df.pivot_table(values='newphoneuser', aggfunc='count', index=x, columns=hue, fill_value=0,
                        margins=True)
    print(da)
    index_tag = [str(i) for i in da.index.values if i != 'All']
    col_tag = [i for i in da.columns.values if i != 'All']
    index_list = list()
    col_list = list()
    value = list()
    for i in index_tag:
        for j in range(0, len(col_tag)):
            index_list.append(i)

    for i in range(0, len(index_tag)):
        col_list.extend(col_tag)

    for i in index_tag:
        i = int(i)
        for j in col_tag:
            percent = round(da.loc[i, j] * 100 / da.loc[i, 'All'], 2)
            value.append(percent)

    show_data = {
        x: index_list,
        hue: col_list,
        'percent': value,
    }
    show_df = pd.DataFrame(show_data)
    ax = sns.lineplot(data=show_df, x=x, y='percent', hue=hue, estimator=None)
    for index, row in show_df.iterrows():
        ax.text(row[x], row['percent'], row['percent'], color='black', ha='center')
    plt.show()


def show_data():
    df = pd.read_csv('../data/telcomcustomer.csv')
    col_names = dict()
    with open('../data/col_name.txt','r',encoding='utf8')as cnf:
        for name in cnf.readlines():
            name = name.split('|')
            col_names[name[0]] = name[1].strip()
    # df=df.drop(columns=['过去三个月的平均月费用','过去六个月的平均月费用','是否流失','客户ID','客户生命周期内平均月费用'])
    df=df.drop(columns=['过去三个月的平均月费用','过去六个月的平均月费用','是否流失','用户id','客户生命周期内平均月费用',
                        '平均已完成呼叫数','平均峰值数据调用次数','已完成语音通话的平均使用分钟数','尝试拨打的平均语音呼叫次数','是否翻新机',
                        '家庭中唯一订阅者的数量','未应答数据呼叫的平均次数','平均完成的语音呼叫数','平均掉线语音呼叫数',
                        '非高峰数据呼叫的平均数量','一分钟内的平均呼入电话数','平均占线数据调用次数','平均尝试调用次数','信息库匹配','平均丢弃数据呼叫数'])
    df.rename(columns=col_names,inplace=True)
    print(len(df.columns))
    userinfo_cols = ['region','marriage_counts','adults_numbers_family','expect_income','has_creditcard',
                     'totalemployed_months','activeusers_family','credit_rating']
    phoneinfo_cols = ['dualband_capability','phoneprice','phonenetwork','newphoneuser','phone_usedays']
    useraction_cols = [col for col in df.columns.values if col not in userinfo_cols and col not in phoneinfo_cols]
    print(len(useraction_cols))
    for col in useraction_cols:
        print(col)
    one_level = df['user_values'].quantile(0.3)
    two_level = df['user_values'].quantile(0.8)
    # def get_value_level(value):
    #     if value<one_level:
    #         value_level =0
    #     elif value>=one_level and value<two_level:
    #         value_level =1
    #     elif value>=two_level:
    #         value_level =2
    #     return value_level
    # df['value_level'] = df['user_values'].apply(lambda x: get_value_level(x))
    #
    #
    # corr_matrix = df[useraction_cols].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Heatmap')
    # plt.show()
    #
    # #
    # values = 'phone_usedays'
    # tt = df.groupby('value_level')[values].apply(lambda x:round(x.mean(),4)).reset_index()
    # ax = sns.barplot(tt,x='value_level',y=values)
    # for index, row in tt.iterrows():
    #     ax.text(row['value_level'], row[values], row[values], color='black', ha='center')
    # plt.show()
    #
    #
    #
    # # x = 'region'
    # # hue = 'value_level'
    # # show_percent(df,x,hue)
    # f
    #
    #  # 交叉验证
    model = xgboost.XGBRegressor(random_state=30)
    target = df['user_values']
    train_data = df.drop(columns='user_values')
    # scorer = make_scorer(r2_score)
    # scores = cross_validate(model,train_data,target,cv=5,n_jobs=-1,return_train_score=True,scoring=scorer)
    # print(f"train {np.mean(scores['train_score'])}|test {np.mean(scores['test_score'])}")
    #
    # f
    # 获取特征重要性
    model.fit(train_data, target)
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    print("特征重要性排名:")
    fealist = []
    feascore = []
    top_feas = list()
    for t in range(len(train_data.columns)):
        top_feas.append(train_data.columns[indices[t]])
        print("%d. 特征名：%s，重要性得分：%f" % (t + 1, train_data.columns[indices[t]], importance[indices[t]]))
        fealist.append(train_data.columns[indices[t]])
        feascore.append(importance[indices[t]])



if __name__ == '__main__':
    show_data()