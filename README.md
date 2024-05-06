## Reproduction
To reproduce our results you need to:
1. Install dependencies - `pip install -r requirements.txt`.
2. To setup data run `./scripts/setup.sh`. The script also downloads precomputed results.
3. To reproduce training of LSTM on all classes run `python bin/train_attention_all_classes.py`.
4. To reproduce training of attention on all classes run `python bin/train_attention_all_classes.py`.
5. To reproduce training of attention on commands run `python bin/train_attention_commands.py`.
6. To reproduce training of attention on main classes `python bin/train_attention_main_classes.py`.
7. To recalculate analysis results run notebook `notebooks/analysis.ipynb`.
