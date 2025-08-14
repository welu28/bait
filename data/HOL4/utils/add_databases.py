import os
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
from kaggle_secrets import UserSecretsClient


def add_databases(data_dir):
    # --- Connect to MongoDB Atlas using Kaggle Secrets ---
    user_secrets = UserSecretsClient()
    uri = user_secrets.get_secret("MONGODB_URI")
    db_client = MongoClient(uri)

    # --- Load all local data files ---
    def load_json(path):
        with open(os.path.join(data_dir, path), "r") as f:
            return json.load(f)

    def load_pickle(path):
        with open(os.path.join(data_dir, path), "rb") as f:
            return pickle.load(f)

    dep_data = load_json("dep_data.json")
    paper_dataset = load_pickle("paper_goals.pk")
    full_db = load_json("new_db.json")
    torch_graph_dict = load_pickle("torch_graph_dict.pk")
    train_test_data = load_pickle("train_test_data.pk")
    token_enc = load_pickle("graph_token_encoder.pk")  # currently unused

    # --- Merge paper goals into full_db ---
    new_db = {v[2]: v for k, v in full_db.items()}
    for goal_id, plain_expr in paper_dataset:
        if goal_id in new_db:
            new_db[goal_id][5] = plain_expr

    # Save adjusted DB
    with open(os.path.join(data_dir, "adjusted_db.json"), "w") as f:
        json.dump(new_db, f)

    # --- Filter only valid goals ---
    valid_goals = [g for g in paper_dataset if g[0] in new_db]
    print(f"Len valid {len(valid_goals)}")
    np.random.shuffle(valid_goals)

    with open(os.path.join(data_dir, "valid_goals_shuffled.pk"), "wb") as f:
        pickle.dump(valid_goals, f)

    # --- MongoDB collections ---
    db = db_client["hol4"]
    dependency_data = db["dependency_data"]
    pretrain_data = db["split_data"]
    paper_split = db["paper_goals"]
    expression_graph_data = db["expression_graphs"]
    vocab = db["vocab"]
    expression_info_data = db["expression_metadata"]

    print(f"Adding HOL4 standard library data to database 'hol4'...\n")

    # --- Insert dependency data (safe upsert) ---
    for k, v in tqdm(dep_data.items(), desc="Dependencies"):
        dependency_data.update_one(
            {"_id": k},
            {"$set": {"dependencies": v}},
            upsert=True
        )

    # --- Insert expression graphs ---
    for k, v in tqdm(torch_graph_dict.items(), desc="Expression Graphs"):
        expression_graph_data.update_one(
            {"_id": k},
            {"$set": {"data": v}},
            upsert=True
        )

    # --- Insert pretrain split data ---
    train, val, test, _ = train_test_data
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for conj, stmt, y in tqdm(split_data, desc=f"{split_name.capitalize()} Split"):
            pretrain_data.insert_one({
                "split": split_name,
                "conj": conj,
                "stmt": stmt,
                "y": y
            })

    # --- Build and insert vocabulary ---
    vocab_dict = {}
    i = 1
    for v in torch_graph_dict.values():
        toks = v['sequence'] + v['tokens']
        for tok in toks:
            if tok not in vocab_dict:
                vocab_dict[tok] = i
                i += 1

    for special in ['VAR', 'VARFUNC', 'UNK']:
        vocab_dict[special] = len(vocab_dict)

    for k, idx in tqdm(vocab_dict.items(), desc="Vocabulary"):
        vocab.update_one({"_id": k}, {"$set": {"index": idx}}, upsert=True)

    with open(os.path.join(data_dir, "vocab.pk"), "wb") as f:
        pickle.dump(vocab_dict, f)

    # --- Insert expression metadata ---
    for k, v in tqdm(new_db.items(), desc="Expression Metadata"):
        expression_info_data.update_one(
            {"_id": k},
            {
                "$set": {
                    "theory": v[0],
                    "name": v[1],
                    "dep_id": v[3],
                    "type": v[4],
                    "plain_expression": v[5]
                }
            },
            upsert=True
        )

    # --- Split and insert paper goals ---
    random.shuffle(valid_goals)
    split_point = int(0.8 * len(valid_goals))
    train_goals = valid_goals[:split_point]
    val_goals = valid_goals[split_point:]

    paper_split.insert_many([{"_id": g[0], "plain": g[1], 'split': 'train'} for g in train_goals])
    paper_split.insert_many([{"_id": g[0], "plain": g[1], 'split': 'val'} for g in val_goals])

    print("Database update complete.")