import matplotlib.pyplot as plt
import numpy as np
# libraries & dataset
import seaborn as sns
import pandas as pd


fig, ax = plt.subplots(1, 1,gridspec_kw={'wspace': 0.4},figsize=(10,6))
# sns.set(font_scale = 10)

data = pd.read_excel('results-final.xlsx')

# sns.set_theme(style="ticks", palette="pastel")
sns.set_theme(style="whitegrid", palette="pastel")

# sns.set(style="darkgrid")

# Load the example tips dataset
# tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
p=sns.boxplot(x="% Irregularity", y="AUROC",
            hue="Model", palette=["m", "g", "r"],
            data=data)
sns.despine(offset=10, trim=True)

# p.set_ylabel("AUROC", fontsize = 20)
# p.set_xlabel("% Irregularity", fontsize = 20)
# p.set_title("Plot", fontsize = 20)
# plt.legend(labels=["COPER","Perceiver","LSTM"], fontsize = 20)

# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=3, frameon=False, fontsize=20)
plt.legend(loc="lower left", frameon=True, fontsize=20)
plt.xlabel('% Irregularity', fontsize=20)
plt.ylabel('AUROC', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig('final_results.pdf',dpi=300,bbox_inches='tight')
