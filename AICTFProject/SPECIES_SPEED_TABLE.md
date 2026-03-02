# Species speed parameters (speed_mult range) by team size

Values from `opponent_params.py`. Each entry is `(s_low, s_high)`; speed_mult is sampled uniformly in that range per episode.

| Species   | 2v2 (x, y) | 3v3 (x, y) | 4v4 (x, y) |
|-----------|------------|------------|------------|
| RUSHER    | 1.05, 1.25 | 1.05, 1.25 | 0.80, 0.90 |
| CAMPER    | 0.80, 1.00 | 0.80, 1.00 | 0.78, 0.88 |
| BALANCED  | 0.90, 1.10 | 0.90, 1.10 | 0.78, 0.88 |

- **2v2 and 3v3** share the same base values (both use `n_agents < 4`).
- **4v4** uses the scaled branch (`n_agents >= 4`) for reduced speed and simpler deception.

Optional LaTeX table for the paper:

```latex
\begin{table}[t]
\centering
\caption{Species speed multiplier ranges $(s_{\min}, s_{\max})$ by team size.}
\label{tab:species-speed}
\begin{tabular}{lccc}
\toprule
\textbf{Species} & \textbf{2v2} & \textbf{3v3} & \textbf{4v4} \\
\midrule
RUSHER  & (1.05, 1.25) & (1.05, 1.25) & (0.80, 0.90) \\
CAMPER  & (0.80, 1.00) & (0.80, 1.00) & (0.78, 0.88) \\
BALANCED & (0.90, 1.10) & (0.90, 1.10) & (0.78, 0.88) \\
\bottomrule
\end{tabular}
\end{table}
```
