import pandas as pd
import pprint as pp
import numpy as np
s = pd.Series([9904312, 3448737, 2890451, 2466052],
              index=["서울", "부산", "인천", "대구"])
pp.pprint(s)
pp.pprint(pd.Series(range(10, 14)))

pp.pprint(s.index)
pp.pprint("서울" in s)
pp.pprint("대전" in s)
for idx, val in s.items():
    pp.pprint("%s = %d" % (idx, val))

s0 = {"서울": 9631482, "부산": 3393191, "인천": 2632035, "대전": 1490158}
pp.pprint(s0)
s2 = pd.Series(s0, index=["부산", "서울", "대전", "인천"])
pp.pprint(s2)
ds = s-s2
pp.pprint(ds)
dsn = ds.notnull()
pp.pprint(dsn)
pp.pprint(ds[dsn])

rs = ds/s2 * 100
pp.pprint(rs[rs.notnull()])
rs["부산"] = 1.63
pp.pprint(rs[rs.notnull()])
del rs["부산"]
pp.pprint(rs[rs.notnull()])

data = {
    "2015": [9904312, 3448737, 2890451, 2466052],
    "2010": [9631482, 3393191, 2632035, 2431774],
    "2005": [9762546, 3512547, 2517680, 2456016],
    "2000": [9853972, 3655437, 2466338, 2473990],
    "지역": ["수도권", "경상권", "수도권", "경상권"],
    "2010-2015 증가율": [0.0283, 0.0163, 0.0982, 0.0141]
}
columns = ["지역", "2015", "2010", "2005", "2000", "2010-2015 증가율"]
index = ["서울", "부산", "인천", "대구"]
df = pd.DataFrame(data, index=index, columns=columns)
pp.pprint(df)
df.index.name = 'city'
df.columns.name = 'year'
pp.pprint(df)
pp.pprint(df['2000'])
pp.pprint(df[['2000']])
pp.pprint(type(df['2000']))
pp.pprint(type(df[['2000']]))
pp.pprint(df)
df['2015+2010'] = df['2015'] + df['2010']
pp.pprint(df)
# pandas의 dataframe은 column먼저 인덱싱한 뒤, row인덱싱 > numpy와 반대
pp.pprint(df[1:3])

data = {
    "국어": [80, 90, 70, 30],
    "영어": [90, 70, 60, 40],
    "수학": [90, 60, 80, 70],
}
columns = ["국어", "영어", "수학"]
index = ["춘향", "몽룡", "향단", "방자"]
df = pd.DataFrame(data, index=index, columns=columns)

pp.pprint(df)
pp.pprint(df["수학"])
pp.pprint(df[['국어', '영어']])
df['평균'] = (df['국어'] + df['영어'] + df['수학']) / 3
pp.pprint(df)
df['영어']['방자'] = 80
df['평균'] = (df['국어'] + df['영어'] + df['수학']) / 3
pp.pprint(df)
pp.pprint(df['춘향':'춘향'])
pp.pprint(type(df['춘향':'춘향']))
pp.pprint(df['향단':'향단'].T['향단'])
pp.pprint(type(df['향단':'향단'].T['향단']))

pp.pprint(pd.read_csv('sample1.csv'))
pp.pprint(pd.read_table('sample3.txt', sep='\s+', skiprows=[1, 3]))

df.to_csv('sample6.csv')

df = pd.DataFrame(np.arange(10, 22).reshape(3, 4),
                  index=["a", "b", "c"],
                  columns=["A", "B", "C", "D"])

pp.pprint(df)
pp.pprint(df['A'] > 12)
pp.pprint(type(df['A'] > 12))
pp.pprint(df.loc[df["A"] > 12, "B"])

import seaborn as sns
titanic = sns.load_dataset('titanic')

np.random.seed(1)
s2 = pd.Series(np.random.randint(6, size=100))
pp.pprint(s2.tail())
pp.pprint(s2.value_counts().sort_index())
pp.pprint(s2.sort_values(ascending=False))
pp.pprint(df.sort_values(by='A'))
pp.pprint(df.sort_values(by=['A', 'B']))

pp.pprint(titanic["sex"].value_counts())
pp.pprint(titanic["age"].value_counts())
pp.pprint(titanic["class"].value_counts())
pp.pprint(titanic["alive"].value_counts())

np.random.seed(1)
df2 = pd.DataFrame(np.random.randint(10, size=(4, 8)))
pp.pprint(df2)
pp.pprint(df2.sum(axis=0))
pp.pprint(df2.sum(axis=1))

df3 = pd.DataFrame({
    'A': [1, 3, 4, 3, 4],
    'B': [2, 3, 1, 2, 3],
    'C': [1, 5, 2, 4, 4]
})

pp.pprint(df3)
pp.pprint(df3['C'].max())
pp.pprint(df3.apply(lambda x: x.max() - x.min(),axis=0))
pp.pprint(df3.apply(lambda x: x.max() - x.min(),axis=1))
# axis가 좀 이상하지 않나? apply의 반복방향은 axis 1일때 axis 0을, 반복방햐이 axis 0일때 axis1을 쓴다
# iterator 방향을 axis로 주는 것이 아니라, 요소방향을 axis라고 준다고 생각하자.

pp.pprint(df3.apply(pd.value_counts))
pp.pprint(df3.apply(pd.value_counts, axis=1))
pp.pprint(df3.apply(pd.value_counts).fillna(0).astype(int))

ages = [0, 2, 10, 21, 23, 37, 31, 61, 20, 41, 32, 100]
bins = [1, 15, 25, 35, 60, 99]
labels = ['미성년자', '청년', '중년', '장년', '노년']
cats = pd.cut(ages, bins, labels=labels)
pp.pprint(cats)
pp.pprint(cats.categories)
pp.pprint(cats.codes)

data = {"ages": ages, "age_cat": cats.categories[cats.codes]}
pp.pprint(data)
df4 = pd.DataFrame(data)
pp.pprint(df4)

data1 = {"ages": ages, "age_cat": cats}
df5 = pd.DataFrame(data1)
pp.pprint(df5)

df6 = pd.DataFrame(ages, columns = ["ages"])
df6["age_cat"] = cats
pp.pprint(df6)

titanic