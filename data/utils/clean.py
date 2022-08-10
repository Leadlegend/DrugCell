from collections import defaultdict

datapath = 'data/drugcell_all.txt'
new_data_path = 'data/drugcell_all_clean.txt'


f = open(datapath, 'r', encoding='utf-8')
g = open(new_data_path, 'w', encoding='utf-8')
data_stats = defaultdict(set)

for line in f:
    cell, drug, auc = line.strip().split('\t')
    index = cell + '\t' + drug
    data_stats[index].add(auc)

print(len(data_stats))
for key, val in data_stats.items():
    aucs = list(val)
    l = len(aucs)
    if l == 1:
        text = key + '\t' + str(aucs[0]) + '\n'
    else:
        auc_avg = [float(x) for x in aucs]
        auc_avg = sum(auc_avg) / l
        text = key + '\t' + str(auc_avg) + '\n'
    g.write(text)

f.close()
g.close()
