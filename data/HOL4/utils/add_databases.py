import random
import numpy as np
from tqdm import tqdm
import json
from pymongo import MongoClient
import pickle
from kaggle_secrets import UserSecretsClient

def add_databases(data_dir):
    user_secrets = UserSecretsClient()
    uri = user_secrets.get_secret("MONGODB_URI")
    db_client = MongoClient(uri)  # <-- atlas db, not localhost

    with open(data_dir + "dep_data.json") as f:
        dep_data = json.load(f)

    with open(data_dir + "paper_goals.pk", "rb") as f:
        paper_dataset = pickle.load(f)

    with open(data_dir + "new_db.json") as f:
        full_db = json.load(f)

    with open(data_dir + "torch_graph_dict.pk", "rb") as f:
        torch_graph_dict = pickle.load(f)

    with open(data_dir + "train_test_data.pk", "rb") as f:
        train_test_data = pickle.load(f)

    with open(data_dir + "graph_token_encoder.pk", "rb") as f:
        token_enc = pickle.load(f)

    new_db = {v[2]: v for k, v in full_db.items()}

    for goal in paper_dataset:
        if goal[0] in new_db:
            new_db[goal[0]][5] = goal[1]

    with open(data_dir + "adjusted_db.json", "w") as f:
        json.dump(new_db, f)

    valid_goals = []
    for goal in paper_dataset:
        if goal[0] in new_db.keys():
            valid_goals.append(goal)

    print(f"Len valid {len(valid_goals)}")
    np.random.shuffle(valid_goals)

    with open(data_dir + "valid_goals_shuffled.pk", "wb") as f:
        pickle.dump(valid_goals, f)

    db_name = "hol4"
    info_name = "expression_metadata"
    dep_name = "dependency_data"
    split_name = "split_data"
    paper_name = "paper_goals"
    expression_graph_name = "expression_graphs"
    vocab_name = "vocab"

    db = db_client[db_name]
    dependency_data = db[dep_name]
    pretrain_data = db[split_name]
    paper_split = db[paper_name]
    expression_graph_data = db[expression_graph_name]
    vocab = db[vocab_name]
    expression_info_data = db[info_name]

    print(f"Adding HOL4 standard library data up to and including \"probabilityTheory\" to database {db_name}\n")

    for k, v in tqdm(dep_data.items()):
        dependency_data.insert_one({"_id": k, "dependencies": v})

    for (k, v) in tqdm(torch_graph_dict.items()):
        expression_graph_data.insert_one({"_id": k, "data": v})

    train, val, test, enc_nodes = train_test_data

    for conj, stmt, y in tqdm(train):
        pretrain_data.insert_one({"split": "train", "conj": conj, "stmt": stmt, "y": y})

    for conj, stmt, y in tqdm(val):
        pretrain_data.insert_one({"split": "val", "conj": conj, "stmt": stmt, "y": y})

    for conj, stmt, y in tqdm(test):
        pretrain_data.insert_one({"split": "test", "conj": conj, "stmt": stmt, "y": y})

    vocab_dict = {}
    i = 1
    for v in torch_graph_dict.values():
        toks = v['sequence'] + v['tokens']
        for tok in toks:
            if tok not in vocab_dict:
                vocab_dict[tok] = i
                i += 1

    vocab_dict['VAR'] = len(vocab_dict)
    vocab_dict['VARFUNC'] = len(vocab_dict)
    vocab_dict['UNK'] = len(vocab_dict)

    for k, v in tqdm(vocab_dict.items()):
        vocab.insert_one({"_id": k, "index": v})

    with open(data_dir + "vocab.pk", "wb") as f:
        pickle.dump(vocab_dict, f)

    for k, v in tqdm(new_db.items()):
        expression_info_data.insert_one(
            {"_id": k, "theory": v[0], "name": v[1], "dep_id": v[3], "type": v[4], "plain_expression": v[5]}
        )

    random.shuffle(valid_goals)
    train_goals = valid_goals[:int(0.8 * len(valid_goals))]
    val_goals = valid_goals[int(0.8 * len(valid_goals)):]

    paper_split.insert_many([{"_id": g[0], "plain": g[1], 'split': 'train'} for g in train_goals])
    paper_split.insert_many([{"_id": g[0], "plain": g[1], 'split': 'val'} for g in val_goals])