import json

import pandas as pd
from flask import Flask, render_template
from pyecharts import options as opts
from pyecharts.charts import Line, Scatter

# 创建Flask对象
app = Flask(__name__)


# 视图
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hotcorr")
def hotcorr():
    corr_df = pd.read_csv('E:\TelcomCustomer_Analysis\data\showdata\hotcorr.csv')
    factor_df = pd.read_csv('E:\TelcomCustomer_Analysis\data\showdata\hotfactor.csv')
    corr_x = corr_df.columns
    corr_y = corr_df.columns
    cor_data = list()
    for col in corr_df.columns:
        for idx, value in corr_df[col].items():
            cor_data.append([corr_df.columns.get_loc(col),idx, round(value, 2)])
    factor_y = factor_df['Unnamed: 0'].values.tolist()
    factor_df.drop(columns='Unnamed: 0', inplace=True)
    factor_x = factor_df.columns
    factor_data = list()
    for col in factor_df.columns:
        for idx, value in factor_df[col].items():
            factor_data.append([factor_df.columns.get_loc(col),idx, round(value, 4)])
    print(factor_data)
    return render_template("hotcorr.html", corr_x=corr_x, corr_y=corr_y, cordata=cor_data,
                           factor_x =factor_x ,factor_y=factor_y, factordata=factor_data)


@app.route("/phoneshow")
def phoneshow():
    df1 = pd.read_csv(r'E:\TelcomCustomer_Analysis\data\showdata\userphonedays.csv')
    df1['phone_usedays'] = round(df1['phone_usedays'], 2)
    one_x = df1['value_level'].values.tolist()
    one_y = df1['phone_usedays'].values.tolist()
    userphonedays = {
        'x': one_x,
        'y': one_y
    }
    df2 = pd.read_csv('E:\TelcomCustomer_Analysis\data\showdata\phoneuseradio.csv')
    usedays_x = df2['phone_usedays_level'].unique().tolist()
    one = df2.loc[df2['value_level'] == 1, 'percent'].values.tolist()
    two = df2.loc[df2['value_level'] == 2, 'percent'].values.tolist()
    three = df2.loc[df2['value_level'] == 3, 'percent'].values.tolist()
    useday_data = {
        'usedays_x': usedays_x,
        'one': one,
        'two': two,
        'three': three
    }
    df3 = pd.read_csv(r'E:\TelcomCustomer_Analysis\data\showdata\useday2priceScatter.csv')
    df3['value_level'] =df3['value_level'].astype(int)
    df3['phone_usedays'] =df3['phone_usedays'].astype(int)
    df3['phoneprice'] =df3['phoneprice'].astype(int)
    a = df3.loc[df3['value_level'] == 1]
    b = df3.loc[df3['value_level'] == 2]
    c = df3.loc[df3['value_level'] == 3]
    a_list = list()
    b_list = list()
    c_list = list()
    for index, row in a.iterrows():
        a_list.append([float(row['phone_usedays']), float(row['phoneprice'])])
    for index, row in b.iterrows():
        b_list.append([float(row['phone_usedays']), float(row['phoneprice'])])
    for index, row in c.iterrows():
        c_list.append([float(row['phone_usedays']), float(row['phoneprice'])])
    scatter_data = {
        'one': a_list,
        'two': b_list,
        'three': c_list,
    }
    return render_template("phoneshow.html",userphonedays = userphonedays,useday_data=useday_data,scatter_data=scatter_data)

@app.route("/familyshow")
def familyshow():
    df1 = pd.read_csv('E:\TelcomCustomer_Analysis\data\showdata\employeduseradio.csv')
    employed_x = df1['employed_level'].unique().tolist()
    one = df1.loc[df1['value_level'] == 1, 'percent'].values.tolist()
    two = df1.loc[df1['value_level'] == 2, 'percent'].values.tolist()
    three = df1.loc[df1['value_level'] == 3, 'percent'].values.tolist()
    employed_data = {
        'employed_x': employed_x,
        'one': one,
        'two': two,
        'three': three
    }
    df2 = pd.read_csv(r'E:\TelcomCustomer_Analysis\data\showdata\familyscatter.csv')
    data = list()
    for index, row in df2.iterrows():
        data.append([float(row['adults_numbers_family']),float(row['employed_level']),round(float(row['value_level_3']),3)])
    scatter_data = {'data':data}
    return render_template("familyshow.html",employed_data = employed_data,scatter_data=scatter_data)

@app.route("/serviceuseshow")
def serviceuseshow():
    df1 = pd.read_csv(r'E:\TelcomCustomer_Analysis\data\showdata\userminutes2useradio.csv')
    useminute_x = df1['useminute_level'].unique().tolist()
    print(useminute_x)
    useminute_x = list(map(lambda x: x * 3, useminute_x))
    print(useminute_x)
    one = df1.loc[df1['value_level'] == 1, 'percent'].values.tolist()
    two = df1.loc[df1['value_level'] == 2, 'percent'].values.tolist()
    three = df1.loc[df1['value_level'] == 3, 'percent'].values.tolist()
    useminute_data = {
        'useminute_x': useminute_x,
        'one': one,
        'two': two,
        'three': three
    }
    return render_template("serviceuseshow.html",useminute_data = useminute_data)

@app.route("/modelshow")
def modelshow():
    df1 = pd.read_csv(r'E:\TelcomCustomer_Analysis\data\showdata\modelparams.csv')
    modeldata = df1.to_dict(orient='records')
    return render_template("modelshow.html",modeldata = modeldata)



if __name__ == '__main__':
    app.run(debug=True)