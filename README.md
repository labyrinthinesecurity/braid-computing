# Knots, Braids and Cloud Permissions

This repository contains the synthetic dataset and the scripts supporting the arxiv preprint called Detecting Privilege Escalation Using Temporal Braid Groups (https://arxiv.org/pdf/2603.10094)  

## Lyapunoy Exponent (LE) calculus
identify_disagreements.py calculates the difference between the abelian firing gate and the non-abelian LE of each of the deployments, input scc_war_ratios.csv output disagreements.txt 

## dataset generation pipeline
If you want to reconstruct the synthetic dataset from scratch:
1. Generate 1000 6-vertices SCCs using burau.py using seed 42, result saved to 6_topologies.json, and generate 49972 deployments for these 1000 SCCs, result saved to scc_war_ratios.csv
2. Generate the 1000 corresponding adjacency matrixes using json2csv.py, result saved to 1000adjacency_matrixes.csv
