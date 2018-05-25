import pandas as pd



describe = pd.read_csv("./eval_utils/%s_aq_describe.csv" % ('bj'))

# print(describe.loc['mean'])
# print(describe.loc['std'])
a = describe.loc[0:1]

print(a)


print(describe.shape)

