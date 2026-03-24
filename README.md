# Knots, Braids and Cloud Permissions

This repository contains the synthetic dataset and the scripts supporting the arxiv preprint called Detecting Privilege Escalation Using Temporal Braid Groups (https://arxiv.org/pdf/2603.10094)  

## Lyapunoy Exponent (LE) calculus
probe.py calculates the LE of a given SCC deployment, input some_deployment.json
identify_disagreements.py calculates the difference between the abelian firing gate and the non-abelian LE of each of the deployments, input scc_war_ratios.csv output disagreements.txt 

## dataset generation pipeline
If you want to reconstruct the synthetic dataset from scratch:
1. Generate 1000 6-vertices SCCs using analyze_scc_collisions.py using seed 42, result saved to 6_topologies.json
2. Generate the corresponding adjacency matrixes using json2csv.py, result saved to 1000adjacency_matrixes.csv
3. Generate the 49972 deployments (based on the above 1000 6-vertices SCCs with 50 random WAR assignments per SCC) using burau.py result saved to scc_war_ratios.csv
