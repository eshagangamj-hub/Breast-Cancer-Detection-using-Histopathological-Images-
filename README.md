How to Run
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Prepare data (download Kaggle Breast Histopathology Images)

# 3) Build graphs (SLIC → superpixel graphs)
python -m src.graphs.build --input data/ --out data/graphs/ --segments 30   # for GCN
python -m src.graphs.build --input data/ --out data/graphs_agcl/ --segments 60  # for AGCL

# 4) Train
python src/train_gcn.py  --data data/graphs/       --epochs 100
python src/train_agcl.py --data data/graphs_agcl/  --epochs 100 --contrastive --edge-mask

# 5) Evaluate
python src/eval.py --ckpt runs/agcl/best.ckpt --data data/graphs_agcl/
