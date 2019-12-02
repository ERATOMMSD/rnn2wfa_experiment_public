import csv
import zipfile
import util_boto3
import os.path
import pickle
import WFA
import json

precalcs = """20190528025718-172-31-27-212_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
20190528025828-172-31-30-230_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
20190528025939-172-31-17-144_precalc_no4_abcdefghijklmnopqrst_15_3.zip
20190528030045-172-31-18-173_precalc_no2_abcdefghijklmno_20_3.zip
20190528025940-172-31-31-24_precalc_no5_abcdefghij_10_3.zip
20190528030106-172-31-20-175_precalc_no5_abcdefghijklmno_20_3.zip
20190528025905-172-31-30-230_precalc_no1_abcdefghijklmnopqrst_20_3.zip
20190528030226-172-31-20-175_precalc_no5_abcdef_10_3.zip
20190528025904-172-31-17-144_precalc_no1_abcdefghij_15_3.zip
20190528030147-172-31-20-175_precalc_no3_abcdefghij_10_3.zip
20190528030014-172-31-17-144_precalc_no3_abcdefghijklmnopqrst_10_3.zip
20190528025748-172-31-18-129_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
20190528025940-172-31-30-230_precalc_no3_abcdefghijklmnopqrst_20_3.zip
20190528025904-172-31-31-24_precalc_no4_abcdefghijklmnopqrst_20_3.zip
20190528030016-172-31-31-24_precalc_no3_abcdefghij_20_3.zip
20190528030050-172-31-17-144_precalc_no4_abcdefghij_10_3.zip
20190528030229-172-31-18-173_precalc_no1_ab_10_3.zip
20190528025940-172-31-27-212_precalc_no4_abcdefghijklmno_20_3.zip
20190528025903-172-31-17-48_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
20190528030200-172-31-18-54_precalc_no2_abcdef_10_3.zip
20190528030050-172-31-31-24_precalc_no3_abcdefghijklmnopqrst_15_3.zip
20190528030050-172-31-27-212_precalc_no5_abcdefghijklmno_15_3.zip
20190528025935-172-31-18-129_precalc_no3_abcdefghijklmno_15_3.zip
20190528030125-172-31-17-144_precalc_no1_abcdefghijklmnopqrst_10_3.zip
20190528030200-172-31-27-212_precalc_no4_abcdef_10_3.zip
20190528030049-172-31-31-254_precalc_no4_abcdefghijklmnopqrst_10_3.zip
20190528025827-172-31-31-254_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
20190528030235-172-31-30-230_precalc_no3_ab_10_3.zip
20190528030050-172-31-17-48_precalc_no3_abcdefghijklmno_20_3.zip
20190528030120-172-31-18-173_precalc_no2_abcdefghijklmno_10_3.zip
20190528025718-172-31-30-230_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
20190528025903-172-31-18-54_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
20190528030015-172-31-31-254_precalc_no1_abcdefghijklmno_20_3.zip
20190528025828-172-31-18-54_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
20190528025718-172-31-31-24_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
20190528025717-172-31-17-48_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
20190528030120-172-31-18-129_precalc_no3_abcdefghijklmno_10_3.zip
20190528025829-172-31-20-175_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
20190528025939-172-31-17-48_precalc_no5_abcdefghijklmnopqrst_10_3.zip
20190528030159-172-31-31-254_precalc_no4_abcd_10_3.zip
20190528025752-172-31-31-254_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
20190528025640-172-31-27-212_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
20190528025713-172-31-18-129_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
20190528025641-172-31-17-48_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
20190528030015-172-31-18-54_precalc_no5_abcdefghij_15_3.zip
20190528030015-172-31-27-212_precalc_no5_abcdefghij_20_3.zip
20190528030230-172-31-18-129_precalc_no2_ab_10_3.zip
20190528025641-172-31-20-175_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
20190528025754-172-31-20-175_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
20190528030235-172-31-18-54_precalc_no5_ab_10_3.zip
20190528025640-172-31-18-173_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
20190528025934-172-31-18-173_precalc_no4_abcdefghij_15_3.zip
20190528025906-172-31-20-175_precalc_no1_abcdefghij_20_3.zip
20190528025640-172-31-31-254_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
20190528025747-172-31-18-173_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
20190528025753-172-31-31-24_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
20190528025753-172-31-17-144_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
20190528025857-172-31-18-173_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
20190528030125-172-31-31-254_precalc_no2_abcdefghij_10_3.zip
20190528030009-172-31-18-173_precalc_no5_abcdefghijklmno_10_3.zip
20190528025717-172-31-31-254_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
20190528030155-172-31-18-129_precalc_no1_abcd_10_3.zip
20190528030125-172-31-17-48_precalc_no2_abcdefghijklmnopqrst_15_3.zip
20190528030045-172-31-18-129_precalc_no5_abcdefghijklmnopqrst_20_3.zip
20190528030126-172-31-27-212_precalc_no4_abcdefghij_20_3.zip
20190528025828-172-31-31-24_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
20190528025717-172-31-17-144_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
20190528025712-172-31-18-173_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
20190528025753-172-31-30-230_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
20190528025940-172-31-18-54_precalc_no1_abcdefghij_10_3.zip
20190528030015-172-31-17-48_precalc_no4_abcdefghijklmno_15_3.zip
20190528030050-172-31-18-54_precalc_no4_abcdefghijklmno_10_3.zip
20190528025822-172-31-18-173_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
20190528025905-172-31-27-212_precalc_no2_abcdefghij_15_3.zip
20190528025717-172-31-18-54_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
20190528030200-172-31-17-144_precalc_no5_abcd_10_3.zip
20190528030200-172-31-17-48_precalc_no3_abcd_10_3.zip
20190528025828-172-31-27-212_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
20190528030200-172-31-31-24_precalc_no3_abcdef_10_3.zip
20190528025827-172-31-17-144_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
20190528025640-172-31-30-230_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
20190528025858-172-31-18-129_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
20190528030200-172-31-30-230_precalc_no2_abcd_10_3.zip
20190528025753-172-31-17-48_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
20190528030126-172-31-18-54_precalc_no1_abcdefghijklmno_10_3.zip
20190528030126-172-31-30-230_precalc_no2_abcdefghij_20_3.zip
20190528030027-172-31-20-175_precalc_no2_abcdefghijklmno_15_3.zip
20190528025902-172-31-31-254_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
20190528030016-172-31-30-230_precalc_no2_abcdefghijklmnopqrst_10_3.zip
20190528030235-172-31-27-212_precalc_no4_ab_10_3.zip
20190528030125-172-31-31-24_precalc_no1_abcdefghijklmnopqrst_15_3.zip
20190528025640-172-31-31-24_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
20190528030010-172-31-18-129_precalc_no5_abcdefghijklmnopqrst_15_3.zip
20190528025640-172-31-17-144_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
20190528025719-172-31-20-175_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
20190528025640-172-31-18-54_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
20190528025753-172-31-18-54_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
20190528025639-172-31-18-129_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
20190528025828-172-31-17-48_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
20190528025946-172-31-20-175_precalc_no3_abcdefghij_15_3.zip
20190528025823-172-31-18-129_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
20190528025753-172-31-27-212_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
20190528030051-172-31-30-230_precalc_no1_abcdefghijklmno_15_3.zip
20190528030154-172-31-18-173_precalc_no1_abcdef_10_3.zip
20190528025939-172-31-31-254_precalc_no2_abcdefghijklmnopqrst_20_3.zip"""
precalcs = precalcs.split("\n")
precalcs = [x.strip() for x in precalcs]

rows = []

for e in precalcs:
    print(e)
    id = int(e.split("_")[2][2])
    util_boto3.download(e)
    with zipfile.ZipFile(e) as z:
        dirname = os.path.splitext(e)[0]
        z.extractall(dirname)
    with open(os.path.join(dirname, "wfa.pickle"), "rb") as f:
        wfa_orig: WFA.WFA = pickle.load(f)
        wfa_orig.callings = set()
        alphabet = wfa_orig.alphabet
        states_orig = wfa_orig.get_size()
    with open(os.path.join(dirname, "result_precalc.json"), "r") as f:
        res_precalc = json.load(f)
        str2rnnvalue = res_precalc["str2rnnvalue"]
      
    n_ok = len([k for k, v in str2rnnvalue.items() if abs(v - wfa_orig.get_value(k)) < 0.05])
    mse = sum([(wfa_orig.get_value(k) - v)**2 for k, v in str2rnnvalue.items()])/len(str2rnnvalue)
    rows.append({"alphabet size": len(alphabet),
                 "original wfa's size": wfa_orig.get_size(),
                 "id": id,
                 "acc005": n_ok/len(str2rnnvalue),
                 "mse": mse
                 })

with open("eval_a2r.csv", "w") as f:
    fieldnames = ["alphabet size", "original wfa's size", "id", "mse", "acc005"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    # for r in rows:
    #     writer.writerow(r)
