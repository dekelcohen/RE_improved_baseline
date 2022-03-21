# .bat - set PYTHONPATH=..\..\nlp-utils\src
export PYTHONPATH=$PYTHONPATH;../../nlp-utils/src; python trainer.py --prepro nyt --data_dir ../Datasets/New York Times Relation Extraction --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed 23 --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base
