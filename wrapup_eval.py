import csv
import zipfile
import util_boto3
import os.path
import pickle
import WFA
import json
import itertools
import numpy as np

eval_perfs = """20191113143008-172-31-10-130_eval_perf_no4_abcdefghijklmno_20_3.zip
20191113143044-172-31-10-130_eval_perf_no1_abcdefghijklmno_10_3.zip
20191113143048-172-31-31-201_eval_perf_no5_abcdef_10_3.zip
20191113143044-172-31-9-196_eval_perf_no1_abcdefghijklmno_15_3.zip
20191113143028-172-31-9-196_eval_perf_no2_abcdefghij_20_3.zip
20191113143049-172-31-10-130_eval_perf_no4_abcdefghij_15_3.zip
20191113143038-172-31-10-130_eval_perf_no1_abcdef_10_3.zip
20191113143038-172-31-26-88_eval_perf_no5_abcd_10_3.zip
20191113143013-172-31-9-196_eval_perf_no3_abcdefghij_20_3.zip
20191113143033-172-31-31-201_eval_perf_no4_abcd_10_3.zip
20191113143024-172-31-10-130_eval_perf_no3_abcdefghij_15_3.zip
20191113143039-172-31-9-196_eval_perf_no4_abcdefghij_20_3.zip
20191113143048-172-31-26-88_eval_perf_no5_abcdefghij_15_3.zip
20191113143024-172-31-26-88_eval_perf_no4_abcdefghijklmno_15_3.zip
20191113143054-172-31-10-130_eval_perf_no5_abcdefghij_10_3.zip
20191113143054-172-31-31-201_eval_perf_no2_abcdefghijklmno_10_3.zip
20191113143009-172-31-31-201_eval_perf_no3_abcdefghijklmno_10_3.zip
20191113143013-172-31-31-201_eval_perf_no2_abcdefghijklmno_20_3.zip
20191113143019-172-31-31-201_eval_perf_no2_abcdefghijklmno_15_3.zip
20191113143009-172-31-9-196_eval_perf_no3_abcdefghijklmno_20_3.zip
20191113143023-172-31-9-196_eval_perf_no4_abcdef_10_3.zip
20191113143019-172-31-9-196_eval_perf_no1_abcdefghij_20_3.zip
20191113143044-172-31-31-201_eval_perf_no1_abcdefghij_15_3.zip
20191113143033-172-31-10-130_eval_perf_no3_abcdefghij_10_3.zip
20191113143008-172-31-26-88_eval_perf_no1_abcdefghijklmno_20_3.zip
20191113143024-172-31-31-201_eval_perf_no4_abcdefghij_10_3.zip
20191113143019-172-31-10-130_eval_perf_no4_abcdefghijklmno_10_3.zip
20191113143053-172-31-9-196_eval_perf_no3_abcd_10_3.zip
20191113143048-172-31-9-196_eval_perf_no1_abcd_10_3.zip
20191113143014-172-31-10-130_eval_perf_no3_abcdefghijklmno_15_3.zip
20191113143053-172-31-26-88_eval_perf_no2_abcdef_10_3.zip
20191113143044-172-31-26-88_eval_perf_no5_abcdefghijklmno_20_3.zip
20191113143029-172-31-31-201_eval_perf_no2_abcdefghij_15_3.zip
20191113143018-172-31-26-88_eval_perf_no3_abcdef_10_3.zip
20191113143034-172-31-26-88_eval_perf_no1_abcdefghij_10_3.zip
20191113143029-172-31-26-88_eval_perf_no5_abcdefghij_20_3.zip
20191113143014-172-31-26-88_eval_perf_no5_abcdefghijklmno_15_3.zip
20191113143038-172-31-31-201_eval_perf_no2_abcd_10_3.zip
20191113143028-172-31-10-130_eval_perf_no2_abcdefghij_10_3.zip
20191113143034-172-31-9-196_eval_perf_no5_abcdefghijklmno_10_3.zip"""
eval_perfs = eval_perfs.split("\n")
eval_perfs = [x.strip() for x in eval_perfs]
print(len(set(eval_perfs)), len(eval_perfs))
input()
rows = []

masaki_tail = ["1-1.zip", "2-2.zip", "3-4.zip", "4-5.zip"]

for e in eval_perfs:
    if "adfa" in e:
        flag_adfa = True
    else:
        flag_adfa = False
    flag_paren = ("paren" in e)

    flag_masaki = any(e.endswith(i) for i in masaki_tail)
    print(e)
    util_boto3.download(e)
    task_name = "_".join(e.split("_")[3:])
    with zipfile.ZipFile(e) as z:
        dirname = os.path.splitext(e)[0]
        z.extractall(dirname)

    if flag_adfa:
        nozip = e[:-4]
        splitted = nozip.split("_")
        proj = splitted[-2]
        alphabet_size = int(splitted[-1])
        import numpy as np

        wfa_orig = WFA.WFA("a", np.array([[0]]), np.array([[0]]), {"a": np.array([[0]])})
        alphabet = "a" * alphabet_size
    elif flag_masaki:
        with open(os.path.join(dirname, "alphabet.tsv"), "r") as f:
            alphabet = "".join(f.readline().split())
        wfa_orig = WFA.WFA("a", np.array([[0]]), np.array([[0]]), {"a": np.array([[0]])})
        alphabet_size = len(alphabet)
    elif flag_paren:
        alphabet = "()0123456789"
        wfa_orig = WFA.WFA("a", np.array([[0]]), np.array([[0]]), {"a": np.array([[0]])})
        alphabet_size = len(alphabet)
    else:
        with open(os.path.join(dirname, "wfa.pickle"), "rb") as f:
            wfa_orig: WFA.WFA = pickle.load(f)
            alphabet = wfa_orig.alphabet
            states_orig = wfa_orig.get_size()
    comment = ""
    with open(os.path.join(dirname, "args_rnn2wfa.json"), "r") as f:
        args = json.load(f)
        method = args["eqq_type"]
        if method == "regr":
            with open(os.path.join(dirname, "eqq_param.json"), "r") as f1:
                eqq_params = json.load(f1)
                experimental_constant_allowance = eqq_params["experimental_constant_allowance"]
                regressor_name = eqq_params["regressor_maker_name"]
                comment = eqq_params["comment"]
        elif method == "search":
            with open(os.path.join(dirname, "eqq_param.json"), "r") as f1:
                eqq_params = json.load(f1)
                comment = "search"
                if eqq_params["experimental_reset"]:
                    comment += "_reset"
                else:
                    comment += "_noreset"
            experimental_constant_allowance = ""
            regressor_name = ""
            comment = "search"
            comment += str(eqq_params["quit_number"])
        else:
            experimental_constant_allowance = ""
            regressor_name = ""

    with open(os.path.join(dirname, "result_eval_perf.json"), "r") as f:
        result_eval_perf = json.load(f)
        wfa_data = result_eval_perf["wfa_data"]
        mse = wfa_data["mse"]
        time_wfa_infer = wfa_data["time_wfa"]
        acc005 = wfa_data["acc005"]
        time_rnn_infer = result_eval_perf["time_rnn"]
        process_wfa = result_eval_perf["process_data"]
        process_acc = []
        for snapshot in process_wfa:
            process_acc.append(snapshot['acc005'])
            # time_and_acc += f"{snapshot['acc005']}@{snapshot['time_wfa']:.2f} - "

    with open(os.path.join(dirname, "statistics.json"), "r") as f:
        stat = json.load(f)
        extracting_time = stat["extracting_time"]
        periods_lstar = stat["periods_lstar"]
        calling_mem = stat["calling_mem"]
        size_wfa = stat["size_wfa"]
        eqq_time = stat["eqq_time"]
        table_time = stat["table_time"]
        add_ce_time = stat["add_ce_time"]
        stats_in_eqq = stat["stats_in_eqq"]
        time_regression = ""
        time_finding_points = ""
        time_calc_min = ""
        if stats_in_eqq is not None:
            for i in stats_in_eqq:
                if i is None:
                    break
                time_regression += f"{i['time_regression']:.1f} - "
                time_finding_points += f"{i['time_finding_points']:.1f} - "
                time_calc_min += f"{i['time_calc_min']:.1f} - "

    zipped = list(itertools.zip_longest(process_acc, periods_lstar))
    process_time_and_acc = " - ".join(f"{a}@{t:.2f}" for a, t in zipped)

    bad_ce = 0
    counterexamples = []
    if not flag_adfa:
        with open(os.path.join(dirname, "rnn2wfa.log"), "r") as f:
            for line in f.readlines():
                if "INFO - Given option is" in line:
                    bad_ce = 0
                    counterexamples = []
                if "bad counterexample" in line:
                    bad_ce += 1
                if "Found a counterexample" in line:
                    counterexamples.append(line.split("'")[1])
    counterexamples = " - ".join(counterexamples)


    def get_time_to_acc(acc):
        l = [t for a, t in zipped if a >= acc]
        if l:
            return l[0]
        else:
            return None


    time_to_accs = [(f"time_to_{x}", get_time_to_acc(x / 10.0)) for x in range(1, 10)]

    process_total = zip(process_wfa, periods_lstar, calling_mem)
    d = {"task name": task_name,
         "alphabet size": len(alphabet),
         "original wfa's size": wfa_orig.get_size(),
         "method": method,
         "comment": comment,
         "experimental_constant_allowance": experimental_constant_allowance,
         "regressor_name": regressor_name,
         "mse": mse,
         "acc005": acc005,
         "extracting_time": extracting_time,
         "timer_rnn_infer": time_rnn_infer,
         "time_wfa_infer": time_wfa_infer,
         "calling memQ": calling_mem[-1],
         "#lstar": len(periods_lstar),
         "extracted wfa's size": size_wfa,
         "time_regression": time_regression,
         "time_finding_points": time_finding_points,
         "time_calc_min": time_calc_min,
         "time_and_acc": process_time_and_acc,
         "file": e,
         "table_time": table_time,
         "add_ce_time": add_ce_time,
         "eqq_time": eqq_time,
         "bad_ce": bad_ce,
         "counterexamples": counterexamples
         # "lstar_loop": lstar_loop
         }
    print(dict(time_to_accs))
    d.update(dict(time_to_accs))
    rows.append(d)

with open("wrapup.csv", "w") as f:
    fieldnames = ["task name", "alphabet size", "original wfa's size", "method", "comment",
                  # "experimental_constant_allowance",
                  # "regressor_name",
                  "mse", "acc005", "extracting_time",
                  "timer_rnn_infer", "time_wfa_infer",
                  "calling memQ", "#lstar", "extracted wfa's size", "table_time", "add_ce_time", "eqq_time",
                  "bad_ce",
                  "time_regression", "time_finding_points", "time_calc_min",
                  "counterexamples",
                  "time_and_acc", "file"] + [x[0] for x in time_to_accs]
    deleting = set(d.keys()) - set(fieldnames)
    for row in rows:
        for i in deleting:
            del row[i]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

with open("wrapup.csv", "r") as f:
    x = "".join(f.readlines())
    x = x.replace("\n\n", "\n")

with open("wrapup.csv", "w") as f:
    f.write(x)
