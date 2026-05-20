# The MES formula

Per epoch, given a target task T:

\[
\begin{aligned}
z_\mu    &= z(\text{ERD}_{\mu,\,\text{contra}}, \text{baseline}_T) \\
z_\beta  &= z(\text{ERD}_{\beta,\,\text{contra}}, \text{baseline}_T) \\
z_{LI}   &= z(\text{LateralizationIndex}, \text{baseline}_T) \\
z_{MRCP} &= z(\text{MRCP}_{\text{amplitude}}, \text{baseline}_T) \\
p_{model}&= \text{calibrated posterior probability of class }T \\
\end{aligned}
\]

\[
\text{raw} = w_1 z_\mu + w_2 z_\beta + w_3 z_{LI} + w_4 z_{MRCP} + w_5 \,\text{logit}(p_{model})
\]

\[
\text{MES} = 100 \cdot \sigma(\text{raw}) \quad \in [0, 100]
\]

## Where the weights come from

Weights \(w_i\) are learned by logistic regression of the per-trial
feature vector \([\,z_\mu, z_\beta, z_{LI}, z_{MRCP}, \text{logit}(p_{model})\,]\)
against **dataset-given task-vs-rest labels** (task=1, rest=0). This is
deliberate: fitting against the classifier's own predictions would be
circular and inflate \(w_5\).

## Per-subject baseline

\(\text{baseline}_T\) is a 4-vector of means and standard deviations,
estimated from a rest block at the start of the session. When no rest
block is available, MES falls back to a population-level baseline (mean=0,
std=1) and clearly labels the report.

## Bounded by construction

The sigmoid ensures MES is always in \([0, 100]\) regardless of how extreme
the input features get. This is critical for clinicians comparing scores
across sessions.
