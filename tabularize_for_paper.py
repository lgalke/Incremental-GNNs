import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("results_csv_file", nargs='+', help="Path to results file")
parser.add_argument("--full", default=False, action='store_true', help="Path to results file")
parser.add_argument("--save", help="Prefix for saving (w/o extension)")

args = parser.parse_args()

df = pd.read_csv(args.results_csv_file[0])
for path in args.results_csv_file[1:]:
    df_next = pd.read_csv(path)
    df = pd.concat([df, df_next])


import numpy as np
print(df.dtypes)
df.rename(columns={'history': 'window'}, inplace=True)
# df.dataset = df.dataset.map({'7dc': 'pharmabio', 'dblp-graph': 'dblp-easy', 'dblp-graph-hard': 'dblp-hard'})
inf_value = 'inf'
df.window = df.window.map(lambda c: inf_value if c >= 20 else str(c))
# df = df[df.start == 'warm']

print("****************")
print("*** Columns: ***", *df.columns, sep='\n')
print("****************")
N = len(df)
print("N", N)
# g = sns.catplot(x='history', y='accuracy', kind='bar', hue='model', data=df)
g = sns.catplot(x='model', y='accuracy', col='dataset', kind='bar', hue='window', data=df)

if not args.save:
    args.save = osp.splitext(args.results_csv_file[0])[0] + '-temporal_aggregation'

window_col = 'wind.'
df.rename(columns={'window': window_col}, inplace=True)
inf_col = 'cmp. inf'

groups = df.groupby(["dataset", "model", window_col, "start"])

df = pd.DataFrame(groups['accuracy'].mean())
df['SD'] = groups['accuracy'].std()
df['SE'] = groups['accuracy'].std() / np.sqrt(groups['accuracy'].count())
# Finish stats
print(df)

if not args.full:
    ix = df.groupby(['dataset', 'model', window_col])['accuracy'].idxmax()
    df = df.loc[ix]
    df.reset_index(level=-1, inplace=True)
    print(df)
    df['acc95'] = df['start'].map(lambda s: '{}: '.format(s[0])) + df['accuracy'].map('{:.3f}'.format).map(lambda s: s.lstrip('0')) + "+-" + (1.96 * df['SE']).map('{:.2f}'.format).map(lambda s: s.lstrip('0'))

    percent_of_inf = []
    for row in df.itertuples():
        percent_of_inf.append(row.accuracy * 100 / df.loc[(row.Index[0], row.Index[1], inf_value)].accuracy)
    df[inf_col] = percent_of_inf
    df[inf_col] = df[inf_col].map('{:3.0f}%'.format)

    df.drop(columns=['SD','SE','accuracy','start'], inplace=True)
    col_fmt = 'llrrrrrr'
else:
    df['acc95'] = df['accuracy'].map('{:.3f}'.format).map(lambda s: s.lstrip('0')) + "+-" + (1.96 * df['SE']).map('{:.2f}'.format).map(lambda s: s.lstrip('0'))
    percent_of_inf = []
    for row in df.itertuples():
        percent_of_inf.append(row.accuracy * 100 / df.loc[(row.Index[0], row.Index[1], inf_value, row.Index[3])].accuracy)
    df[inf_col] = percent_of_inf
    df[inf_col] = df[inf_col].map('{:3.0f}%'.format)

    df.drop(columns=['SD','SE','accuracy'], inplace=True)
    col_fmt = 'lllrrrrrr'

df.rename(columns={'acc95':'avg. acc.'}, inplace=True)

df = df.unstack(level=1)
df = df.swaplevel(axis=1)
df.sort_index(axis=1,level=[0,1],inplace=True)
latex_table = df.to_latex(multirow=True,
                               bold_rows=True,
                               # col_space=5,
                               column_format=col_fmt,
                               caption=f"""
                               Average accuracy across seeds and time steps with varying temporal window sizes, 95\% confidence intervals are computed based on sample variance.
                               We only list the best performing values of cold (c) and warm (w) restarts for each configuration.
                               We compare each average accuracy to the average with the full graph available in the column '{inf_col}'. N={N}""",
                               multicolumn_format='c',
                               label="tab:results")
latex_table = latex_table.replace('+-', '$\pm$')
latex_table = latex_table.replace('inf', '$\infty$')
print(latex_table)
print(df)

print("Saving table to", args.save + '.tex') 
with open(args.save + '.tex', 'w') as fh:
    print(latex_table, file=fh)



print("Saving figure to", args.save + '.png') 
plt.savefig(args.save +  '.png')

        
