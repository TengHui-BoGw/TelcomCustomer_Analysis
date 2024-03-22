import datetime
from base_utils import *
import pandas as pd

def create_table():
    user_info = """
                    CREATE TABLE IF NOT EXISTS `user_info` (
                        user_id INT,
                        region INT,
                        marriage_counts INT,
                        adults_numbers_family INT,
                        expect_income INT,
                        has_creditcard INT,
                        totalemployed_months INT,
                        activeusers_family INT,
                        credit_rating INT
                        )
                    """
    phone_info = """
        CREATE TABLE IF NOT EXISTS `phone_info` (
                user_id INT,
                dualband_capability INT,
                phonenetwork INT,
                newphoneuser INT,
                phone_usedays INT,
                phoneprice INT
            )
        """

    service_useage_info = """
        CREATE TABLE IF NOT EXISTS `service_useage_info` (
    	user_id INT, 
    	user_values INT,
    	useminutes INT,
    	over_useminutes INT,
    	over_cost INT,
    	voicecost INT,
    	overdata_cost INT,
    	roaming_callcounts INT,
    	useminutes_percentchange_before_threemonth INT,
    	cost_percentchange_before_threemonth INT,
    	buzy_callcounts INT,
    	miss_callcounts INT,
    	try_usedata_counts INT,
    	answercounts INT,
    	complete_usedata_counts INT,
    	customerservice_callcounts INT,
    	customerservice_useminutes INT,
    	conferencecall_counts INT,
    	inAndout_callcounts_PVC INT,
    	incomplete_minutes_PVC INT,
    	callcounts_NPVC INT,
    	drop_callcounts INT,
    	forward_callcounts INT,
    	wait_callcounts INT,
    	user_spend_limit INT,
    	total_callcounts_lifecycle INT,
    	total_useminutes_lifecycle INT,
    	totalcost_lifecycle INT,
    	totalcost_billadjust INT,
    	totalminutes_billadjust INT,
    	callcounts_billadjust INT,
    	avg_useminutes_monthly INT,
    	callcounts_monthly INT,
    	avg_useminutes_before_threemonth INT,
    	avg_callcounts_before_threemonth INT,
    	avg_useminutes_before_sixmonth INT,
    	avg_callcounts_before_sixmonth INT
    	)
        """
    sql_list = list()
    sql_list.append(user_info)
    sql_list.append(phone_info)
    sql_list.append(service_useage_info)
    conn = mysql_db()
    try:
        with conn.cursor() as cursor:
            for sql in sql_list:
                cursor.execute(sql)
                conn.commit()
                print("Table created successfully!")
    finally:
        conn.close()

def load_csv2mysql():
    col_names = dict()
    with open('../data/col_name.txt', 'r', encoding='utf8') as cnf:
        for name in cnf.readlines():
            name = name.split('|')
            col_names[name[0]] = name[1].strip()
    conn = mysql_db()
    df = pd.read_csv(r'../data/telcomcustomer.csv')
    df = df.drop(
        columns=['过去三个月的平均月费用', '过去六个月的平均月费用', '是否流失', '客户生命周期内平均月费用',
                 '平均已完成呼叫数', '平均峰值数据调用次数', '已完成语音通话的平均使用分钟数',
                 '尝试拨打的平均语音呼叫次数', '是否翻新机',
                 '家庭中唯一订阅者的数量', '未应答数据呼叫的平均次数', '平均完成的语音呼叫数', '平均掉线语音呼叫数',
                 '非高峰数据呼叫的平均数量', '一分钟内的平均呼入电话数', '平均占线数据调用次数', '平均尝试调用次数',
                 '信息库匹配', '平均丢弃数据呼叫数'])
    df.rename(columns=col_names, inplace=True)
    user_infofields = ['user_id','region','marriage_counts','adults_numbers_family','expect_income','has_creditcard',
                     'totalemployed_months','activeusers_family','credit_rating']
    phone_infofields = ['user_id','dualband_capability','phoneprice','phonenetwork','newphoneuser','phone_usedays']
    service_useageinfofields =['user_id','user_values', 'useminutes', 'over_useminutes', 'over_cost', 'voicecost', 'overdata_cost',
                            'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                            'cost_percentchange_before_threemonth', 'buzy_callcounts', 'miss_callcounts', 'try_usedata_counts',
                            'answercounts', 'complete_usedata_counts', 'customerservice_callcounts', 'customerservice_useminutes',
                            'conferencecall_counts', 'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                            'drop_callcounts', 'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                            'total_callcounts_lifecycle', 'total_useminutes_lifecycle', 'totalcost_lifecycle',
                            'totalcost_billadjust', 'totalminutes_billadjust', 'callcounts_billadjust', 'avg_useminutes_monthly',
                            'callcounts_monthly', 'avg_useminutes_before_threemonth', 'avg_callcounts_before_threemonth',
                            'avg_useminutes_before_sixmonth', 'avg_callcounts_before_sixmonth']
    userinfo_data = df[user_infofields]
    phoneinfo_data = df[phone_infofields]
    serviceuseage_data = df[service_useageinfofields]
    load_userinfodata =list()
    load_phoneinfodata =list()
    load_serviceuseagedata =list()
    for index1, row1 in userinfo_data.iterrows():
        load_userinfodata.append(tuple(row1))
    for index2, row2 in phoneinfo_data.iterrows():
        load_phoneinfodata.append(tuple(row2))
    for index3, row3 in serviceuseage_data.iterrows():
        load_serviceuseagedata.append(tuple(row3))
    insert_userinfo = f"""
                INSERT INTO user_info({','.join(user_infofields)}) 
                VALUES (%s, %s, %s, %s, %s,
                        %s, %s, %s, %s)
                """
    insert_phoneinfo = f"""
            INSERT INTO phone_info({','.join(phone_infofields)}) 
                        VALUES (%s, %s, %s, %s, %s, %s)
            """
    insert_serviceuseageinfo = f"""
            INSERT INTO service_useage_info({','.join(service_useageinfofields)}) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s)
            """
    sqldata_list = list()
    sqldata_list.append((insert_userinfo,load_userinfodata))
    sqldata_list.append((insert_phoneinfo,load_phoneinfodata))
    sqldata_list.append((insert_serviceuseageinfo,load_serviceuseagedata))
    try:
        with conn.cursor() as cursor:
            for i in sqldata_list:
                print(datetime.datetime.now())
                print('开始导入数据......')
                cursor.executemany(i[0],i[1])
                print('Load Succfully!!!')
            conn.commit()
    finally:
        conn.close()
if __name__ == '__main__':
    pass
    # create_table()
    # load_csv2mysql()