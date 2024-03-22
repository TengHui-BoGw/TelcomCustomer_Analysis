import pymysql


def mysql_db():
    # 连接数据库
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='cth17724006813',
        database='telcomcustomer',
        charset='utf8mb4',  # 设置字符集，确保支持中文等特殊字符
        cursorclass=pymysql.cursors.DictCursor  # 设置游标类型，使查询结果以字典形式返回
    )
    if conn.open:
        return conn
    else:
        print('连接数据库失败')
        raise Exception