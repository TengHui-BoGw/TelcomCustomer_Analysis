import pandas as pd

# 创建一个示例 DataFrame
data = {'value': [5, 15, 25, 35, 10, 20, 30, 40]}
df = pd.DataFrame(data)

# 指定分组的区间
bins = [i for i in range(0,max(df['value']),10)]
print(bins)
f

# 使用cut函数进行分组
df['group'] = pd.cut(df['value'], bins=bins, labels=False)

print(df)
