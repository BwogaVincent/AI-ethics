# AI-ethics
# COMPAS Dataset Bias Analysis Report

## Overview
The COMPAS Recidivism Dataset was analyzed for racial bias in risk scores using IBM's AI Fairness 360 toolkit. The analysis focused on disparities between African-American (unprivileged) and Caucasian (privileged) groups, with metrics like Disparate Impact, Statistical Parity Difference, and False Positive Rate (FPR) Difference.

## Findings
- **Disparate Impact**: Initial value of {di:.2f}, indicating bias as values < 1 suggest unfavorable outcomes for African-Americans compared to Caucasians.
- **Statistical Parity Difference**: Initial value of {spd:.2f}, showing a higher likelihood of high-risk scores for African-Americans.
- **False Positive Rate Difference**: A value of {fpr_diff:.2f}, highlighting that African-Americans are more likely to be falsely labeled as high-risk compared to Caucasians.

These metrics confirm findings from prior studies, such as ProPublica’s 2016 analysis, which noted African-Americans were nearly twice as likely to be incorrectly labeled high-risk.[](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)

## Remediation
To mitigate bias, the Reweighing algorithm was applied, adjusting instance weights to balance group representation. Post-reweighing metrics showed:
- **Disparate Impact**: Improved to {di_transf:.2f}, closer to 1, indicating reduced bias.
- **Statistical Parity Difference**: Reduced to {spd_transf:.2f}, suggesting more equitable risk score distributions.
- **False Positive Rate Difference**: Unchanged, as reweighing primarily addresses group-level disparities.

## Visualizations
A bar plot comparing metrics before and after reweighing is saved as 'compas_fairness_metrics.png', illustrating improvements in fairness metrics.

## Recommendations
While reweighing reduced some bias, persistent FPR disparities suggest further mitigation, such as adversarial debiasing or post-processing methods like Equalized Odds. Additionally, addressing systemic biases in data collection (e.g., arrest rates) is critical, as algorithms may perpetuate existing inequities. Regular audits and transparent reporting are recommended to ensure fairness in deployment.[](https://massivesci.com/articles/machine-learning-compas-racism-policing-fairness/)