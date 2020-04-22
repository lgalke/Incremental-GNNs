import argparse
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("results_csv_file", help="Path to results file")
parser.add_argument("-g", "--groupby", nargs='+', help="Group on this variable",
                    default=['model', 'annual_lr'])
parser.add_argument("--latex", default=False, action='store_true', help="Produce latex output")
parser.add_argument("--save", help="Path to save resulting table")
args = parser.parse_args()
print("Loading data:", args.results_csv_file)
df = pd.read_csv(args.results_csv_file)
print("N =", len(df))
print("Grouping by:", args.groupby)
groups = df.groupby(args.groupby)
results = pd.DataFrame(groups['accuracy'].mean())
results['SD'] = groups['accuracy'].std()
results['SE'] = groups['accuracy'].std() / groups['accuracy'].count()
results['acc-ci95'] = results['accuracy'].map('{:.4f}'.format) + "+-" + (1.96 * results['SE']).map('{:.2f}'.format)
print(results)
if args.save:
    print("Saving aggregated results to:", args.save)
    if args.latex:
        with open(args.save, 'w') as fhandle:
            print(results.to_latex(), fhandle)
    else:
        results.to_csv(args.save)
