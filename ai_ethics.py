import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Load and preprocess COMPAS dataset
compas_data = CompasDataset()
privileged_groups = [{'race': 1}]  # Caucasian
unprivileged_groups = [{'race': 0}]  # African-American

# Calculate fairness metrics
metric = BinaryLabelDatasetMetric(compas_data, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)

# Disparate Impact and Statistical Parity
di = metric.disparate_impact()
spd = metric.statistical_parity_difference()

# Apply reweighing to mitigate bias
reweighing = Reweighing(unprivileged_groups=unprivileged_groups,
                       privileged_groups=privileged_groups)
compas_transf = reweighing.fit_transform(compas_data)

# Metrics after reweighing
metric_transf = BinaryLabelDatasetMetric(compas_transf,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
di_transf = metric_transf.disparate_impact()
spd_transf = metric_transf.statistical_parity_difference()

# Classification metrics for false positive rates
class_metric = ClassificationMetric(compas_data, compas_transf,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
fpr_diff = class_metric.difference(class_metric.false_positive_rate)

# Visualization
plt.figure(figsize=(10, 6))
metrics = ['Disparate Impact', 'Stat. Parity Diff.', 'FPR Difference']
before = [di, spd, fpr_diff]
after = [di_transf, spd_transf, fpr_diff]
x = range(len(metrics))
plt.bar(x, before, width=0.45, label='Before Reweighing', alpha=0.6)
plt.bar([i + 0.45 for i in x], after, width=0.45, label='After Reweighing', alpha=0.6)
plt.xticks([i + 0.225 for i in x], metrics)
plt.ylabel('Metric Value')
plt.title('Fairness Metrics Before and After Reweighing')
plt.legend()
plt.tight_layout()
plt.savefig('compas_fairness_metrics.png')
plt.close()
