import os
import pickle
import yaml
import time
from datetime import timedelta
from xmlc.tree_utils import index_tree
from treelib import Tree


def build_label_tree(labels, n=0):
    # create tree and add root node
    tree = Tree()
    root = tree.create_node("Root", "Root")
    # use a flat label tree (i.e. no label grouping at all)
    if n == 0:
        # add each label as a direct child of the root node
        for label in labels:
            tree.create_node(label, label, parent=root)
    else:
        # group the labels using their first n characters
        label_groups = {}
        for label in labels:
            # get the label group
            group = label[:n]
            # check if there already exists a node for that group
            if group not in label_groups:
                group_node = tree.create_node(group, group, parent=root)
                label_groups[group] = group_node
            # add the label to the group
            group_node = label_groups[group]
            tree.create_node(label, label, parent=group_node)

    return tree


if __name__ == '__main__':

    print("Start building label tree")
    t0 = time.time()

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Build the label tree.")
    parser.add_argument("--output-file", type=str,
                        help="File to save the label tree at.")

    # parser arguments
    args = parser.parse_args()
    output_path = os.path.dirname(args.output_file)

    # load preprocessing parameters
    with open(f"{output_path}/params.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)['label_tree']

    # group the labels by their first n characters
    # e.g. 3-0355.1 -> 3-0 for n=3
    n = params['group_id_chars']

    # load labels
    with open(os.path.join(params['label_file_path'], 'labels_vocab.txt'), "r") as f:
        labels = f.read().splitlines()

    # build the label tree and index it
    tree = build_label_tree(labels, n=n)
    tree = index_tree(tree)

    # save the tree
    with open(args.output_file, "wb+") as f:
        pickle.dump(tree, f)

    print(
        f"Building label tree completed completed. Elapsed time: {str(timedelta(seconds=time.time() - t0)).split('.', 2)[0]}")
