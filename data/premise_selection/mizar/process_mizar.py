import glob
import torch
import random
import pickle
import os
from tqdm import tqdm

from data.utils.graph_data_utils import get_depth_from_graph, get_directed_edge_index
from .ast_def_mizar import goal_to_graph, graph_to_dict

if __name__ == '__main__':
    add_attention = False
    file_dir = 'raw_data'
    out_dir = 'mizar_pickles'  # <- directory where we save all .pkl files
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(file_dir + '/*')

    expression_dict = {}
    mizar_labels = []

    for file in tqdm(files, desc="Processing raw files"):
        pos_thms, neg_thms = [], []

        with open(file) as f:
            lines = f.readlines()

        assert lines[0][0] == 'C', f"File {file} does not start with a conjecture (C)"

        for line in lines:
            if line[0] == 'C':
                conj = line[1:].strip("\n")
                if conj not in expression_dict:
                    expression_dict[conj] = graph_to_dict(goal_to_graph(conj))

            elif line[0] == '-':
                neg_thm = line[1:].strip("\n")
                neg_thms.append(neg_thm)
                if neg_thm not in expression_dict:
                    expression_dict[neg_thm] = graph_to_dict(goal_to_graph(neg_thm))

            elif line[0] == '+':
                pos_thm = line[1:].strip("\n")
                pos_thms.append(pos_thm)
                if pos_thm not in expression_dict:
                    expression_dict[pos_thm] = graph_to_dict(goal_to_graph(pos_thm))

            else:
                raise Exception(f"Invalid line in {file}: {line}")

        mizar_labels.append((conj, pos_thms, neg_thms))

    random.shuffle(mizar_labels)
    train_data = mizar_labels[:int(0.8 * len(mizar_labels))]
    val_data = mizar_labels[int(0.8 * len(mizar_labels)):int(0.9 * len(mizar_labels))]
    test_data = mizar_labels[int(0.9 * len(mizar_labels)):]

    def build_pairs(dataset):
        pairs = []
        for conj, pos_thms, neg_thms in dataset:
            pairs.extend([(conj, p, 1) for p in pos_thms])
            pairs.extend([(conj, n, 0) for n in neg_thms])
        return pairs

    train_pairs = build_pairs(train_data)
    val_pairs = build_pairs(val_data)
    test_pairs = build_pairs(test_data)

    vocab = {}
    idx = 0
    for k in expression_dict.keys():
        polished_goal = [c for c in k.split(" ") if c not in ['', '\n']]
        expression_dict[k]['sequence'] = polished_goal
        for tok in polished_goal:
            if tok not in vocab:
                vocab[tok] = idx + 1  # reserve 0 for padding
                idx += 1

    vocab['VAR'] = len(vocab)
    vocab['VARFUNC'] = len(vocab)

    if add_attention:
        print("Adding attention edge index and depth to graphs...")
        for k, v in tqdm(expression_dict.items()):
            attention_edge_index = get_directed_edge_index(
                len(v['tokens']),
                torch.LongTensor(v['edge_index'])
            ).tolist()

            depth = get_depth_from_graph(
                len(v['tokens']),
                torch.LongTensor(v['edge_index'])
            ).tolist()

            v['attention_edge_index'] = attention_edge_index
            v['depth'] = depth

    with open(os.path.join(out_dir, "expr_dict.pkl"), "wb") as f:
        pickle.dump(expression_dict, f)

    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_pairs, f)

    with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_pairs, f)

    with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_pairs, f)

    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    print(f"Saved preprocessed data to directory: {out_dir}")