from graphviz import Digraph
import WFA
import util
import argparse
import util_boto3
import zipfile
import os.path
import pickle

colors = ["red", "blue", "green"]


def make_node_id(wfa, i):
    q0 = wfa.q0.reshape((-1,))
    final = wfa.final.reshape((-1,))
    return f"[{i}]/{q0[i]:.2e}/{final[i]:.2e}"


def depict_wfa(wfa: WFA.WFA):
    G = Digraph(format="png")
    print(f"# q0")
    print(wfa.q0)
    print(f"# final")
    print(wfa.final)
    for i in range(wfa.get_size()):
        G.node(make_node_id(wfa, i))
    for aid, a in enumerate(wfa.alphabet):
        for i in range(wfa.get_size()):
            for j in range(wfa.get_size()):
                if abs(wfa.delta[a][i,j]) > 0.01 and a in ["(", ")"]:
                    G.edge(make_node_id(wfa, i), make_node_id(wfa, j), f"{a}/{wfa.delta[a][i, j]:.2e}",
                        color=colors[aid % len(colors)])
        print(f"# delta[{a}]")
        print(wfa.delta[a])
    fn = util.get_time_hash()
    G.render(fn)
    return fn + ".png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Depict WFA')
    parser.add_argument('filename',
                        help='filename of pikcle')
    parser.add_argument('--s3', type=str, default="",
                        help='target zip file if it exists')
    args = parser.parse_args()
    if args.s3 != "":
        util_boto3.download(args.s3)
        with zipfile.ZipFile(args.s3) as z:
            z.extractall("temp_depict_wfa")
        filepath = os.path.join("temp_depict_wfa", args.filename)
    else:
        filepath = args.filename
    with open(filepath, "rb") as f:
        wfa: WFA.WFA = pickle.load(f)
        x = depict_wfa(wfa)
    print(x)
