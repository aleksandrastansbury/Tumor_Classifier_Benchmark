"""
run_experiments.py : runs Experiments 1-4 over seeds 0-9 and saves one CSV per experiment in results/.
Exp 1 (H1) feature count: top 5, 10, 15, all 30 by ANOVA F-score 
Exp 2 (H2) KNN distances: euclidean, manhattan, chebyshev
Exp 3 (H3) unpruned trees: ID3, C4.5, CART 
Exp 4 probability cutoffs: 0.20 to 0.50 
Plz Use get_split() from data.py, models from model.py, and model_eval from metrics.py !!!!!!
"""