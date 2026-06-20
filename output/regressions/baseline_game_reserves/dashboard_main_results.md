# Baseline game-reserve run dashboard

Leggi questo file prima delle singole tabelle. Le tabelle dettagliate sono in `main/` e `appendix/`.

## Controlli

Main controls: `hh_size, share_female, n_children_u15, n_adults_15plus`.
`head_age` è robustness, non main, per evitare attrition nascosta del campione.

## Righe chiave

| section | outcome | control_specification | coef | se | p_value | q_value_bh | approx_log_effect_pct | n | n_clusters | share_dropped_missing_model_vars | cell_n_treated_proxy_post | interpretation_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_average_welfare_25km | cons_food_share_real |  | 0.0021 | 0.008 | 0.795 | 0.8489 |  | 8860 | 692 | 0.0126 |  | weak_or_null |
| main_average_welfare_25km | log_cons_ae |  | -0.0632 | 0.0344 | 0.0662 | 0.2423 | -6.1242 | 8860 | 692 | 0.0126 |  | nominal_only |
| main_average_welfare_25km | log_cons_pc |  | -0.0637 | 0.0355 | 0.0728 | 0.2423 | -6.1691 | 8860 | 692 | 0.0126 |  | nominal_only |
| main_maasai_expanded_heterogeneity_25km | cons_food_share_real |  | 0.0075 | 0.0241 | 0.7566 | 0.8191 |  | 8860 | 692 | 0.0126 | 57.0 | weak_or_null |
| main_maasai_expanded_heterogeneity_25km | log_cons_ae |  | -0.4381 | 0.2617 | 0.0941 | 0.4745 | -35.475 | 8860 | 692 | 0.0126 | 57.0 | nominal_only |
| main_maasai_expanded_heterogeneity_25km | log_cons_pc |  | -0.4414 | 0.2969 | 0.1371 | 0.4799 | -35.6853 | 8860 | 692 | 0.0126 | 57.0 | weak_or_null |
| key_control_robustness_head_age_and_shocks | cons_food_share_real | main_no_head_age | 0.0021 | 0.008 | 0.795 | 0.795 |  | 8860 | 692 | 0.0126 |  | weak_or_null |
| key_control_robustness_head_age_and_shocks | cons_food_share_real | plus_head_age | 0.003 | 0.0078 | 0.6968 | 0.7601 |  | 6779 | 563 | 0.2445 |  | weak_or_null |
| key_control_robustness_head_age_and_shocks | cons_food_share_real | plus_head_age_and_shocks | 0.0059 | 0.0081 | 0.4672 | 0.623 |  | 6619 | 563 | 0.2623 |  | weak_or_null |
| key_control_robustness_head_age_and_shocks | cons_food_share_real | plus_shocks | 0.0051 | 0.0084 | 0.5396 | 0.6475 |  | 8700 | 692 | 0.0304 |  | weak_or_null |
| key_control_robustness_head_age_and_shocks | log_cons_ae | main_no_head_age | -0.0632 | 0.0344 | 0.0662 | 0.1867 | -6.1242 | 8860 | 692 | 0.0126 |  | nominal_only |
| key_control_robustness_head_age_and_shocks | log_cons_ae | plus_head_age | -0.0465 | 0.0324 | 0.151 | 0.2454 | -4.544 | 6779 | 563 | 0.2445 |  | weak_or_null |
| key_control_robustness_head_age_and_shocks | log_cons_ae | plus_head_age_and_shocks | -0.0591 | 0.0344 | 0.0864 | 0.1867 | -5.7348 | 6619 | 563 | 0.2623 |  | nominal_only |
| key_control_robustness_head_age_and_shocks | log_cons_ae | plus_shocks | -0.0746 | 0.0366 | 0.0414 | 0.1867 | -7.1928 | 8700 | 692 | 0.0304 |  | nominal_only |
| key_control_robustness_head_age_and_shocks | log_cons_pc | main_no_head_age | -0.0637 | 0.0355 | 0.0728 | 0.1867 | -6.1691 | 8860 | 692 | 0.0126 |  | nominal_only |
| key_control_robustness_head_age_and_shocks | log_cons_pc | plus_head_age | -0.0463 | 0.0332 | 0.1636 | 0.2454 | -4.5198 | 6779 | 563 | 0.2445 |  | weak_or_null |
| key_control_robustness_head_age_and_shocks | log_cons_pc | plus_head_age_and_shocks | -0.0592 | 0.0353 | 0.0934 | 0.1867 | -5.7437 | 6619 | 563 | 0.2623 |  | nominal_only |
| key_control_robustness_head_age_and_shocks | log_cons_pc | plus_shocks | -0.0756 | 0.0377 | 0.0448 | 0.1867 | -7.2855 | 8700 | 692 | 0.0304 |  | nominal_only |
| appendix_weighted_welfare_25km | cons_food_share_real | main_no_head_age | 0.0019 | 0.01 | 0.8476 | 0.8476 |  | 8860 | 692 | 0.0126 |  | weak_or_null |
| appendix_weighted_welfare_25km | log_cons_ae | main_no_head_age | -0.0727 | 0.0375 | 0.0529 | 0.1062 | -7.0097 | 8860 | 692 | 0.0126 |  | nominal_only |
| appendix_weighted_welfare_25km | log_cons_pc | main_no_head_age | -0.0701 | 0.0388 | 0.0708 | 0.1062 | -6.7707 | 8860 | 692 | 0.0126 |  | nominal_only |

## Warning automatici

- Almeno una specificazione perde più del 20% del campione dopo il filtro missing (26.2%).
- La cella treated × Maasai-expanded proxy × post è piccola in almeno una riga dashboard: controlla table_0 e table_3.
