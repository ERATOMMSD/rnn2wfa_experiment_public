from flask import Flask, make_response, request, abort
import pickle
import threading
import requests
import datetime
import hashlib
import time
import sys
import logging
import pathlib
import copy
import os
import itertools
import json
import subprocess
import eval_perf # ERASEHERE
import util
from typing import *


def get_time_hash():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


# State of worker.  It can be ("idle",), ("working", task), ("done", task, res) or
lock_state = threading.Lock()
state_worker = ("idle",)

# ------ User write below ------
# Hostnames are written separated by breaks.
# hosts = """
# 999.999.999.999
# 999.999.999.998
# 999.999.999.997
# """
# The following try-except block reads hostnames from hosts.txt.
# If you wrote the hostnames above, please disable the block.
try:
    hosts = pathlib.Path("hosts.txt").read_text()
except FileNotFoundError:
    hosts = ""
ports = [8080, ]

name = "sample"
interval_polling = 5
timeout = 30


def make_list(s: str) -> List[str]:
    xs = s.split("\n")
    xs = [x.strip() for x in xs]
    return xs


# Returns the list of the tasks
def make_tasks():
    extracteds= """20191028222030-172-31-21-154_extract_no3_abcdefghijklmno_10_3.zip
20191029010709-172-31-23-69_extract_no2_abcdefghijklmno_10_3.zip
20191028193346-172-31-17-2_extract_no4_abcdefghijklmno_20_3.zip
20191028203922-172-31-26-224_extract_no3_abcd_10_3.zip
20191028222026-172-31-8-126_extract_no3_abcdefghijklmno_20_3.zip
20191028193337-172-31-9-88_extract_no5_abcdefghij_20_3.zip
20191028193339-172-31-18-27_extract_no5_abcdefghij_10_3.zip
20191028193339-172-31-6-228_extract_no5_abcdefghijklmno_10_3.zip
20191029044610-172-31-23-200_extract_no1_abcdefghij_20_3.zip
20191029044609-172-31-22-241_extract_no1_abcdefghij_10_3.zip
20191028222025-172-31-1-206_extract_no4_abcdefghij_10_3.zip
20191028194842-172-31-25-96_extract_no4_abcdefghijklmno_10_3.zip
20191029044610-172-31-19-149_extract_no1_abcdefghij_10_3.zip
20191028195313-172-31-16-145_extract_no4_abcdefghijklmno_10_3.zip
20191029010713-172-31-8-126_extract_no2_abcdefghij_15_3.zip
20191028200409-172-31-16-94_extract_no4_abcdefghij_20_3.zip
20191028200404-172-31-6-228_extract_no4_abcdef_10_3.zip
20191028193812-172-31-10-247_extract_no4_abcd_10_3.zip
20191028193338-172-31-9-237_extract_no5_abcdefghij_15_3.zip
20191028193340-172-31-21-123_extract_no5_abcdefghij_10_3.zip
20191028222407-172-31-17-2_extract_no2_abcd_10_3.zip
20191029044611-172-31-18-196_extract_no1_abcdefghijklmno_20_3.zip
20191028193340-172-31-21-154_extract_no4_abcdefghijklmno_15_3.zip
20191028193337-172-31-23-69_extract_no4_abcdefghijklmno_20_3.zip
20191028222030-172-31-23-121_extract_no3_abcdefghijklmno_10_3.zip
20191029011144-172-31-10-247_extract_no1_abcdefghijklmno_15_3.zip
20191029024606-172-31-21-86_extract_no1_abcdef_10_3.zip
20191028222028-172-31-3-110_extract_no3_abcdefghijklmno_15_3.zip
20191028223411-172-31-7-175_extract_no2_abcdef_10_3.zip
20191028183353-172-31-5-169_extract_no5_abcdefghij_15_3.zip
20191028193340-172-31-3-110_extract_no5_abcdefghijklmno_20_3.zip
20191029044609-172-31-21-129_extract_no2_abcdefghijklmno_10_3.zip
20191028170155-172-31-25-96_extract_no5_abcd_10_3.zip
20191029000724-172-31-5-169_extract_no2_abcdefghijklmno_15_3.zip
20191028222028-172-31-9-237_extract_no3_abcdefghijklmno_15_3.zip
20191028193344-172-31-23-121_extract_no4_abcdefghijklmno_15_3.zip
20191028193339-172-31-1-206_extract_no5_abcdefghijklmno_15_3.zip
20191028170624-172-31-16-145_extract_no5_abcd_10_3.zip
20191029012213-172-31-25-96_extract_no1_abcdefghijklmno_10_3.zip
20191028193340-172-31-10-247_extract_no5_abcdefghij_20_3.zip
20191028193339-172-31-8-135_extract_no5_abcdefghijklmno_10_3.zip
20191028193339-172-31-8-126_extract_no5_abcdefghijklmno_20_3.zip
20191029044609-172-31-30-240_extract_no1_abcdefghij_20_3.zip
20191028223118-172-31-18-27_extract_no3_abcdefghij_15_3.zip
20191029024335-172-31-25-209_extract_no1_abcdef_10_3.zip
20191029010715-172-31-21-154_extract_no2_abcdefghij_10_3.zip
20191028222031-172-31-17-2_extract_no3_abcdefghij_20_3.zip
20191028222031-172-31-7-175_extract_no3_abcdefghij_20_3.zip
20191029005411-172-31-1-206_extract_no2_abcdefghij_20_3.zip
20191028203158-172-31-9-88_extract_no4_abcdef_10_3.zip
20191028194434-172-31-18-27_extract_no4_abcd_10_3.zip
20191028222049-172-31-6-228_extract_no3_abcdef_10_3.zip
20191028222458-172-31-10-247_extract_no3_abcdefghij_15_3.zip
20191029010714-172-31-23-121_extract_no2_abcdefghij_15_3.zip
20191029044609-172-31-23-210_extract_no1_abcdefghij_15_3.zip
20191028231526-172-31-3-110_extract_no2_abcdef_10_3.zip
20191029004028-172-31-16-94_extract_no2_abcdefghijklmno_15_3.zip
20191028172444-172-31-16-94_extract_no5_abcdef_10_3.zip
20191029013423-172-31-25-96_extract_no1_abcd_10_3.zip
20191028231843-172-31-9-88_extract_no2_abcdefghijklmno_20_3.zip
20191028222024-172-31-23-69_extract_no4_abcdefghij_15_3.zip
20191028223529-172-31-25-96_extract_no3_abcdefghij_10_3.zip
20191028222025-172-31-8-135_extract_no4_abcdefghij_10_3.zip
20191028222330-172-31-6-228_extract_no2_abcd_10_3.zip
20191028232608-172-31-26-224_extract_no2_abcdefghijklmno_20_3.zip
20191029021748-172-31-24-151_extract_no1_abcd_10_3.zip
20191028223959-172-31-16-145_extract_no3_abcdefghij_10_3.zip
20191028222027-172-31-21-123_extract_no3_abcdefghijklmno_20_3.zip
20191029010714-172-31-21-123_extract_no2_abcdefghij_10_3.zip
20191028212039-172-31-5-169_extract_no4_abcdefghij_15_3.zip
20191028171936-172-31-26-224_extract_no5_abcdef_10_3.zip
20191029010712-172-31-9-237_extract_no2_abcdefghij_20_3.zip
20191028200622-172-31-26-224_extract_no4_abcdefghij_20_3.zip
20191028202718-172-31-26-224_extract_no3_abcd_10_3.zip
20191029044609-172-31-21-16_extract_no1_abcdefghij_15_3.zip
20191029044610-172-31-21-32_extract_no1_abcdefghijklmno_20_3.zip
20191029011806-172-31-18-27_extract_no1_abcdefghijklmno_15_3.zip
20191029012057-172-31-7-175_extract_no1_abcdefghijklmno_10_3.zip
20191028193341-172-31-7-175_extract_no5_abcdefghijklmno_15_3.zip
20191028215342-172-31-16-94_extract_no3_abcdef_10_3.zip"""
#     extracteds = """20190830164649-172-31-27-60_extract_no4_abcdefghij_15_3.zip
# 20190830091423-172-31-25-112_extract_no1_abcdefghij_10_3.zip
# 20190830114010-172-31-30-70_extract_no5_abcdefghij_20_3.zip
# 20190830122321-172-31-18-182_extract_no4_abcdefghijklmno_15_3.zip
# 20190830103937-172-31-18-223_extract_no1_abcdefghijklmno_20_3.zip
# 20190830092942-172-31-19-112_extract_no2_abcdefghij_20_3.zip
# 20190830130014-172-31-18-47_extract_no4_abcdef_10_3.zip
# 20190830123725-172-31-24-230_extract_no1_abcdef_10_3.zip
# 20190830153233-172-31-21-81_extract_no1_abcdefghijklmno_10_3.zip
# 20190830114200-172-31-31-115_extract_no3_abcdefghij_20_3.zip
# 20190830090953-172-31-23-90_extract_no5_abcdef_10_3.zip
# 20190830130801-172-31-18-223_extract_no3_abcdefghijklmno_10_3.zip
# 20190830153017-172-31-24-121_extract_no4_abcdefghijklmno_10_3.zip
# 20190830093426-172-31-18-17_extract_no1_abcdefghij_20_3.zip
# 20190830121313-172-31-18-47_extract_no4_abcdefghijklmno_20_3.zip
# 20190830161654-172-31-28-231_extract_no4_abcdefghijklmno_10_3.zip
# 20190830164813-172-31-25-124_extract_no4_abcdefghij_15_3.zip
# 20190830104241-172-31-30-185_extract_no4_abcdefghijklmno_15_3.zip
# 20190830140958-172-31-27-110_extract_no3_abcdefghijklmno_20_3.zip
# 20190830145309-172-31-18-47_extract_no3_abcdefghijklmno_20_3.zip
# 20190830133141-172-31-19-180_extract_no2_abcdefghijklmno_15_3.zip
# 20190830121912-172-31-24-230_extract_no2_abcd_10_3.zip
# 20190830091852-172-31-23-90_extract_no4_abcdefghij_10_3.zip
# 20190830111110-172-31-19-133_extract_no5_abcdefghij_15_3.zip
# 20190830112702-172-31-26-54_extract_no5_abcdefghij_10_3.zip
# 20190830092605-172-31-25-112_extract_no2_abcdefghij_20_3.zip
# 20190830154332-172-31-22-3_extract_no4_abcdefghij_15_3.zip
# 20190830124711-172-31-28-231_extract_no3_abcdef_10_3.zip
# 20190830103844-172-31-31-115_extract_no1_abcdefghij_10_3.zip
# 20190830104349-172-31-28-175_extract_no4_abcdefghij_10_3.zip
# 20190830104017-172-31-31-173_extract_no5_abcdefghij_10_3.zip
# 20190830100720-172-31-27-110_extract_no5_abcdefghijklmno_10_3.zip
# 20190830093857-172-31-18-223_extract_no1_abcdefghij_20_3.zip
# 20190830150449-172-31-29-75_extract_no3_abcdefghijklmno_20_3.zip
# 20190830122138-172-31-19-112_extract_no1_abcdef_10_3.zip
# 20190830124826-172-31-28-231_extract_no3_abcdef_10_3.zip
# 20190830091122-172-31-18-47_extract_no2_abcdef_10_3.zip
# 20190830092541-172-31-26-61_extract_no2_abcdefghij_20_3.zip
# 20190830111041-172-31-23-42_extract_no1_abcdefghijklmno_20_3.zip
# 20190830155324-172-31-29-101_extract_no4_abcdefghij_15_3.zip
# 20190830114226-172-31-20-165_extract_no5_abcd_10_3.zip
# 20190830090923-172-31-23-195_extract_no5_abcdef_10_3.zip
# 20190830092316-172-31-27-60_extract_no1_abcdefghij_15_3.zip
# 20190830102007-172-31-27-42_extract_no1_abcdefghijklmno_20_3.zip
# 20190830123828-172-31-18-182_extract_no2_abcdefghij_10_3.zip
# 20190830143545-172-31-24-49_extract_no3_abcdefghij_15_3.zip
# 20190830130011-172-31-27-110_extract_no4_abcdefghij_20_3.zip
# 20190830152708-172-31-31-173_extract_no1_abcdefghijklmno_10_3.zip
# 20190830121208-172-31-24-230_extract_no2_abcdefghijklmno_20_3.zip
# 20190830100554-172-31-22-119_extract_no2_abcdefghij_15_3.zip
# 20190830132539-172-31-30-185_extract_no1_abcd_10_3.zip
# 20190830120245-172-31-19-133_extract_no4_abcdefghijklmno_20_3.zip
# 20190830143008-172-31-18-17_extract_no4_abcd_10_3.zip
# 20190830104344-172-31-20-198_extract_no5_abcdefghij_20_3.zip
# 20190830133631-172-31-25-124_extract_no1_abcd_10_3.zip
# 20190830123050-172-31-22-196_extract_no2_abcdefghij_15_3.zip
# 20190830100524-172-31-23-90_extract_no2_abcdefghij_20_3.zip
# 20190830132744-172-31-23-90_extract_no1_abcdefghijklmno_15_3.zip
# 20190830124811-172-31-29-75_extract_no2_abcdefghij_10_3.zip
# 20190830134134-172-31-29-75_extract_no3_abcdefghijklmno_10_3.zip
# 20190830141854-172-31-21-81_extract_no3_abcdefghijklmno_20_3.zip
# 20190830141000-172-31-24-230_extract_no4_abcd_10_3.zip
# 20190830093413-172-31-17-80_extract_no1_abcdefghij_20_3.zip
# 20190830100338-172-31-24-49_extract_no4_abcdefghij_10_3.zip
# 20190830144935-172-31-19-133_extract_no1_abcdefghijklmno_15_3.zip
# 20190830102604-172-31-26-61_extract_no3_abcdefghij_10_3.zip
# 20190830133033-172-31-20-198_extract_no2_abcdefghijklmno_20_3.zip
# 20190830110033-172-31-23-90_extract_no5_abcdefghij_15_3.zip
# 20190830133633-172-31-31-115_extract_no4_abcdefghij_20_3.zip
# 20190830145300-172-31-23-195_extract_no1_abcdefghijklmno_15_3.zip
# 20190830113428-172-31-19-180_extract_no3_abcdefghij_20_3.zip
# 20190830100857-172-31-23-42_extract_no5_abcdefghijklmno_20_3.zip
# 20190830105002-172-31-23-90_extract_no1_abcdefghijklmno_20_3.zip
# 20190830140124-172-31-25-124_extract_no3_abcdefghijklmno_15_3.zip
# 20190830143113-172-31-22-196_extract_no4_abcd_10_3.zip
# 20190830104535-172-31-22-119_extract_no1_abcdefghijklmno_20_3.zip
# 20190830091538-172-31-19-112_extract_no2_abcdef_10_3.zip
# 20190830124458-172-31-22-196_extract_no3_abcdef_10_3.zip
# 20190830140330-172-31-24-230_extract_no3_abcd_10_3.zip
# 20190830110958-172-31-27-60_extract_no5_abcdefghijklmno_10_3.zip
# 20190830091052-172-31-28-175_extract_no2_abcdef_10_3.zip
# 20190830093013-172-31-19-133_extract_no5_abcdefghij_20_3.zip
# 20190830103915-172-31-24-230_extract_no3_abcdefghij_10_3.zip
# 20190830123208-172-31-18-223_extract_no1_abcdef_10_3.zip
# 20190830132213-172-31-24-121_extract_no3_abcdefghij_20_3.zip
# 20190830162746-172-31-23-42_extract_no4_abcdefghij_15_3.zip
# 20190830145046-172-31-17-80_extract_no4_abcdefghij_15_3.zip
# 20190830132642-172-31-25-124_extract_no1_abcdefghijklmno_15_3.zip
# 20190830152154-172-31-23-96_extract_no3_abcdefghijklmno_20_3.zip
# 20190830112526-172-31-18-17_extract_no3_abcdefghij_20_3.zip
# 20190830134814-172-31-23-96_extract_no1_abcdefghijklmno_15_3.zip
# 20190830161341-172-31-22-119_extract_no4_abcdefghijklmno_10_3.zip
# 20190830152311-172-31-26-54_extract_no1_abcdefghijklmno_10_3.zip
# 20190830121542-172-31-28-157_extract_no2_abcd_10_3.zip
# 20190830152132-172-31-31-115_extract_no3_abcdefghijklmno_15_3.zip
# 20190830113316-172-31-27-23_extract_no5_abcdefghij_10_3.zip
# 20190830121251-172-31-25-112_extract_no5_abcdefghijklmno_10_3.zip
# 20190830120140-172-31-23-90_extract_no5_abcdefghij_15_3.zip
# 20190830095904-172-31-27-23_extract_no1_abcdefghij_20_3.zip
# 20190830134212-172-31-18-47_extract_no5_abcdefghijklmno_15_3.zip
# 20190830091305-172-31-26-61_extract_no5_abcdef_10_3.zip
# 20190830151935-172-31-24-230_extract_no1_abcdefghijklmno_10_3.zip
# 20190830114153-172-31-30-185_extract_no4_abcdefghijklmno_20_3.zip
# 20190830121134-172-31-24-50_extract_no2_abcdefghij_15_3.zip
# 20190830094247-172-31-18-47_extract_no2_abcdefghij_20_3.zip
# 20190830115747-172-31-31-173_extract_no2_abcdefghijklmno_20_3.zip
# 20190830135653-172-31-17-80_extract_no3_abcd_10_3.zip
# 20190830125936-172-31-28-157_extract_no4_abcdef_10_3.zip
# 20190830093918-172-31-22-119_extract_no5_abcdefghij_20_3.zip
# 20190830151737-172-31-18-182_extract_no3_abcdefghij_15_3.zip
# 20190830095902-172-31-31-173_extract_no5_abcdefghijklmno_20_3.zip
# 20190830124741-172-31-22-196_extract_no3_abcdef_10_3.zip
# 20190830110934-172-31-28-231_extract_no5_abcdefghij_10_3.zip
# 20190830131201-172-31-30-185_extract_no4_abcdefghij_20_3.zip
# 20190830100914-172-31-18-47_extract_no2_abcdefghij_15_3.zip
# 20190830092114-172-31-23-195_extract_no4_abcdefghij_10_3.zip
# 20190830135726-172-31-23-42_extract_no2_abcdefghijklmno_10_3.zip
# 20190830121522-172-31-27-23_extract_no5_abcd_10_3.zip
# 20190830123729-172-31-29-101_extract_no1_abcdef_10_3.zip
# 20190830135101-172-31-29-101_extract_no3_abcdefghij_15_3.zip
# 20190830152619-172-31-30-185_extract_no3_abcdefghijklmno_15_3.zip
# 20190830135956-172-31-17-80_extract_no3_abcd_10_3.zip
# 20190830135648-172-31-27-60_extract_no2_abcdefghijklmno_10_3.zip
# 20190830134908-172-31-17-80_extract_no4_abcdefghijklmno_10_3.zip
# 20190830092336-172-31-27-110_extract_no2_abcdef_10_3.zip
# 20190830143632-172-31-18-223_extract_no5_abcdefghijklmno_15_3.zip
# 20190830163253-172-31-28-157_extract_no3_abcdefghijklmno_20_3.zip
# 20190830103527-172-31-24-121_extract_no2_abcdefghij_15_3.zip
# 20190830090427-172-31-19-112_extract_no2_abcdef_10_3.zip
# 20190830124840-172-31-18-223_extract_no4_abcdef_10_3.zip
# 20190830134813-172-31-24-230_extract_no3_abcdefghij_15_3.zip
# 20190830090633-172-31-22-196_extract_no5_abcdef_10_3.zip
# 20190830094402-172-31-22-196_extract_no4_abcdefghij_10_3.zip
# 20190830141214-172-31-18-17_extract_no4_abcdefghijklmno_20_3.zip
# 20190830132654-172-31-22-119_extract_no1_abcd_10_3.zip
# 20190830132942-172-31-17-80_extract_no5_abcdefghijklmno_15_3.zip
# 20190830093507-172-31-29-75_extract_no2_abcdefghij_20_3.zip
# 20190830114424-172-31-28-104_extract_no5_abcdefghijklmno_20_3.zip
# 20190830094629-172-31-24-50_extract_no1_abcdefghij_10_3.zip
# 20190830121329-172-31-18-223_extract_no2_abcdefghijklmno_20_3.zip
# 20190830120706-172-31-29-101_extract_no4_abcdefghij_20_3.zip
# 20190830103135-172-31-18-17_extract_no4_abcdefghijklmno_15_3.zip
# 20190830115901-172-31-23-96_extract_no3_abcdefghij_20_3.zip
# 20190830123007-172-31-18-47_extract_no2_abcd_10_3.zip
# 20190830123220-172-31-30-70_extract_no4_abcdefghij_20_3.zip
# 20190830121817-172-31-29-101_extract_no2_abcd_10_3.zip
# 20190830113209-172-31-26-61_extract_no3_abcdefghij_10_3.zip
# 20190830161721-172-31-20-198_extract_no4_abcdefghijklmno_10_3.zip
# 20190830123636-172-31-24-49_extract_no5_abcdefghij_10_3.zip
# 20190830114533-172-31-22-3_extract_no5_abcd_10_3.zip
# 20190830113731-172-31-27-110_extract_no5_abcd_10_3.zip
# 20190830095636-172-31-26-54_extract_no5_abcdefghij_20_3.zip
# 20190830144035-172-31-27-42_extract_no2_abcdefghijklmno_15_3.zip
# 20190830143341-172-31-30-70_extract_no1_abcdefghijklmno_10_3.zip
# 20190830125904-172-31-19-112_extract_no2_abcdefghij_10_3.zip
# 20190830133438-172-31-30-70_extract_no5_abcdefghijklmno_15_3.zip
# 20190830141332-172-31-24-230_extract_no4_abcd_10_3.zip
# 20190830121223-172-31-28-157_extract_no4_abcdefghijklmno_20_3.zip
# 20190830102843-172-31-23-96_extract_no1_abcdefghij_15_3.zip
# 20190830100754-172-31-18-223_extract_no2_abcdefghij_15_3.zip
# 20190830110837-172-31-22-119_extract_no5_abcdefghij_15_3.zip
# 20190830104949-172-31-28-157_extract_no1_abcdefghij_20_3.zip
# 20190830122850-172-31-22-3_extract_no2_abcdefghijklmno_15_3.zip
# 20190830125418-172-31-24-50_extract_no3_abcdefghij_15_3.zip
# 20190830123955-172-31-28-231_extract_no2_abcdefghijklmno_10_3.zip
# 20190830152306-172-31-19-180_extract_no3_abcdefghijklmno_15_3.zip
# 20190830131441-172-31-28-231_extract_no3_abcdefghijklmno_10_3.zip
# 20190830120154-172-31-25-124_extract_no4_abcdefghij_20_3.zip
# 20190830115349-172-31-27-42_extract_no3_abcdefghij_10_3.zip
# 20190830143110-172-31-28-104_extract_no2_abcdefghijklmno_15_3.zip
# 20190830134603-172-31-28-157_extract_no5_abcdefghijklmno_15_3.zip
# 20190830132340-172-31-22-119_extract_no2_abcdefghijklmno_10_3.zip
# 20190830141427-172-31-27-110_extract_no4_abcd_10_3.zip
# 20190830154104-172-31-24-50_extract_no3_abcdefghijklmno_10_3.zip
# 20190830123140-172-31-27-23_extract_no1_abcdef_10_3.zip
# 20190830120614-172-31-23-195_extract_no5_abcdefghijklmno_10_3.zip
# 20190830103120-172-31-19-180_extract_no1_abcdefghij_15_3.zip
# 20190830123031-172-31-28-157_extract_no1_abcdef_10_3.zip
# 20190830134045-172-31-30-185_extract_no1_abcd_10_3.zip
# 20190830130528-172-31-17-80_extract_no2_abcdefghijklmno_10_3.zip
# 20190830141442-172-31-30-70_extract_no3_abcdefghijklmno_15_3.zip
# 20190830161434-172-31-23-90_extract_no4_abcdefghijklmno_10_3.zip
# 20190830141005-172-31-22-196_extract_no4_abcd_10_3.zip
# 20190830090214-172-31-28-175_extract_no2_abcdef_10_3.zip
# 20190830112041-172-31-30-185_extract_no2_abcdefghijklmno_20_3.zip
# 20190830121633-172-31-19-112_extract_no5_abcdefghijklmno_10_3.zip
# 20190830122152-172-31-29-75_extract_no4_abcdefghijklmno_15_3.zip
# 20190830141350-172-31-26-54_extract_no4_abcdefghijklmno_20_3.zip
# 20190830123753-172-31-24-50_extract_no2_abcdefghij_10_3.zip
# 20190830091236-172-31-29-75_extract_no5_abcdef_10_3.zip
# 20190830142915-172-31-20-165_extract_no2_abcdefghijklmno_15_3.zip
# 20190830104813-172-31-19-133_extract_no4_abcdefghijklmno_15_3.zip
# 20190830124731-172-31-22-196_extract_no3_abcdef_10_3.zip
# 20190830091057-172-31-24-49_extract_no1_abcdefghij_10_3.zip
# 20190830101940-172-31-27-110_extract_no1_abcdefghijklmno_20_3.zip
# 20190830100703-172-31-27-42_extract_no5_abcdefghijklmno_20_3.zip
# 20190830124016-172-31-22-196_extract_no4_abcdef_10_3.zip
# 20190830110523-172-31-17-80_extract_no4_abcdefghijklmno_15_3.zip
# 20190830151320-172-31-25-112_extract_no2_abcdefghij_10_3.zip
# 20190830134214-172-31-21-81_extract_no2_abcdefghijklmno_10_3.zip
# 20190830110323-172-31-21-81_extract_no1_abcdefghij_20_3.zip
# 20190830122632-172-31-25-112_extract_no2_abcd_10_3.zip
# 20190830135755-172-31-22-3_extract_no2_abcdefghij_10_3.zip
# 20190830160542-172-31-28-175_extract_no3_abcdefghijklmno_15_3.zip
# 20190830101450-172-31-26-61_extract_no5_abcdefghijklmno_10_3.zip
# 20190830114038-172-31-29-101_extract_no5_abcdefghijklmno_20_3.zip
# 20190830130436-172-31-30-70_extract_no4_abcdef_10_3.zip
# 20190830143417-172-31-27-23_extract_no3_abcdefghij_15_3.zip
# 20190830143247-172-31-26-61_extract_no2_abcdefghijklmno_15_3.zip
# 20190830151711-172-31-27-110_extract_no1_abcdefghijklmno_10_3.zip
# 20190830090542-172-31-22-3_extract_no5_abcdef_10_3.zip
# 20190830135841-172-31-29-101_extract_no3_abcd_10_3.zip
# 20190830095445-172-31-28-231_extract_no1_abcdefghij_10_3.zip
# 20190830123857-172-31-27-23_extract_no4_abcdef_10_3.zip
# 20190830111527-172-31-18-47_extract_no3_abcdefghij_10_3.zip
# 20190830134435-172-31-27-110_extract_no5_abcdefghijklmno_15_3.zip
# 20190830135621-172-31-29-101_extract_no3_abcd_10_3.zip
# 20190830133039-172-31-28-175_extract_no2_abcdefghijklmno_20_3.zip
# 20190830142440-172-31-31-173_extract_no1_abcdefghijklmno_15_3.zip
# 20190830114110-172-31-25-124_extract_no1_abcdefghij_15_3.zip
# 20190830101451-172-31-24-230_extract_no1_abcdefghij_10_3.zip
# 20190830105530-172-31-27-110_extract_no3_abcdefghij_20_3.zip
# 20190830094839-172-31-22-3_extract_no4_abcdefghij_10_3.zip
# 20190830114516-172-31-19-180_extract_no5_abcd_10_3.zip
# 20190830111320-172-31-28-157_extract_no5_abcdefghij_15_3.zip
# 20190830132813-172-31-24-121_extract_no1_abcd_10_3.zip
# 20190830112707-172-31-20-165_extract_no5_abcdefghijklmno_20_3.zip
# 20190830113241-172-31-27-110_extract_no5_abcdefghij_15_3.zip
# 20190830093633-172-31-18-182_extract_no5_abcdefghij_20_3.zip
# 20190830124724-172-31-22-196_extract_no3_abcdef_10_3.zip
# 20190830122239-172-31-24-50_extract_no2_abcd_10_3.zip
# 20190830154552-172-31-19-112_extract_no3_abcdefghijklmno_10_3.zip
# 20190830093724-172-31-24-121_extract_no1_abcdefghij_15_3.zip
# 20190830140126-172-31-22-196_extract_no3_abcdefghijklmno_10_3.zip
# 20190830103438-172-31-27-110_extract_no3_abcdefghij_10_3.zip
# 20190830114601-172-31-26-61_extract_no5_abcd_10_3.zip
# 20190830133007-172-31-28-231_extract_no1_abcd_10_3.zip
# 20190830113232-172-31-22-3_extract_no5_abcdefghij_10_3.zip
# 20190830140005-172-31-27-60_extract_no3_abcd_10_3.zip
# 20190830093234-172-31-30-185_extract_no1_abcdefghij_15_3.zip"""
    #     extracteds = """20190819152052-172-31-25-201_extract_no1_abcdefghijklmno_15_3.zip
    # 20190819180740-172-31-9-58_extract_no5_abcd_10_3.zip
    # 20190819152051-172-31-4-76_extract_no2_abcdefghij_15_3.zip
    # 20190819152052-172-31-6-24_extract_no4_abcdefghijklmno_20_3.zip
    # 20190819180738-172-31-23-53_extract_no5_abcdefghijklmno_20_3.zip
    # 20190819152051-172-31-10-205_extract_no4_abcdefghij_10_3.zip
    # 20190819152052-172-31-9-93_extract_no4_abcdefghijklmno_15_3.zip
    # 20190819180737-172-31-18-188_extract_no5_abcdefghijklmno_15_3.zip
    # 20190819180740-172-31-7-64_extract_no1_abcdefghijklmno_10_3.zip
    # 20190819152051-172-31-7-64_extract_no1_abcdefghij_10_3.zip
    # 20190819152051-172-31-26-198_extract_no4_abcdefghij_15_3.zip
    # 20190819152051-172-31-0-60_extract_no5_abcdefghij_15_3.zip
    # 20190819180741-172-31-26-198_extract_no4_abcdef_10_3.zip
    # 20190819152050-172-31-18-188_extract_no3_abcdefghij_20_3.zip
    # 20190819180741-172-31-3-101_extract_no1_abcd_10_3.zip
    # 20190819180740-172-31-6-24_extract_no2_abcdefghijklmno_10_3.zip
    # 20190819180739-172-31-2-117_extract_no3_abcdefghijklmno_10_3.zip
    # 20190819152052-172-31-6-83_extract_no2_abcdefghijklmno_20_3.zip
    # 20190819180739-172-31-1-191_extract_no2_abcdefghij_10_3.zip
    # 20190819180742-172-31-0-60_extract_no4_abcd_10_3.zip
    # 20190819180738-172-31-10-205_extract_no4_abcdefghijklmno_10_3.zip
    # 20190819180740-172-31-25-201_extract_no2_abcdef_10_3.zip
    # 20190819180739-172-31-11-247_extract_no2_abcdefghij_20_3.zip
    # 20190819152051-172-31-1-93_extract_no5_abcdefghijklmno_10_3.zip
    # 20190819180739-172-31-1-93_extract_no3_abcd_10_3.zip
    # 20190819152050-172-31-23-53_extract_no5_abcdefghij_20_3.zip
    # 20190819152051-172-31-9-58_extract_no5_abcdefghij_10_3.zip
    # 20190819152051-172-31-11-247_extract_no3_abcdefghij_15_3.zip
    # 20190819180739-172-31-9-93_extract_no2_abcd_10_3.zip
    # 20190819152052-172-31-2-117_extract_no1_abcdefghijklmno_20_3.zip
    # 20190819152051-172-31-13-126_extract_no3_abcdefghijklmno_15_3.zip
    # 20190819180739-172-31-13-126_extract_no4_abcdefghij_20_3.zip
    # 20190819152052-172-31-1-191_extract_no2_abcdefghijklmno_15_3.zip
    # 20190819180738-172-31-10-132_extract_no1_abcdef_10_3.zip
    # 20190819152051-172-31-10-132_extract_no1_abcdefghij_15_3.zip
    # 20190819180741-172-31-6-83_extract_no5_abcdef_10_3.zip
    # 20190819180739-172-31-4-76_extract_no3_abcdefghij_10_3.zip
    # 20190819152052-172-31-15-244_extract_no3_abcdefghijklmno_20_3.zip
    # 20190819152050-172-31-3-101_extract_no1_abcdefghij_20_3.zip
    # 20190819180741-172-31-15-244_extract_no3_abcdef_10_3.zip"""

    extracteds = make_list(extracteds)
    #sorted
    precalcs = """20190830180152-172-31-31-158_precalc_no3_abcdefghijklmno_10_3.zip
20190830180328-172-31-31-158_precalc_no4_abcd_10_3.zip
20190830175929-172-31-29-220_precalc_no5_abcdefghij_15_3.zip
20190830175632-172-31-31-158_precalc_no5_abcdefghij_20_3.zip
20190830180335-172-31-29-220_precalc_no1_abcdefghijklmno_10_3.zip
20190830175634-172-31-29-220_precalc_no1_abcdefghij_15_3.zip
20190830175840-172-31-4-107_precalc_no1_abcdefghijklmno_20_3.zip
20190830175839-172-31-31-158_precalc_no5_abcdefghij_10_3.zip
20190830180155-172-31-4-107_precalc_no5_abcdefghijklmno_15_3.zip
20190830180230-172-31-29-220_precalc_no3_abcdefghijklmno_15_3.zip
20190904183404-172-31-26-241_precalc_no3_abcdef_10_3.zip
20190830175807-172-31-4-107_precalc_no5_abcdefghijklmno_10_3.zip
20190830180301-172-31-4-107_precalc_no4_abcdefghij_15_3.zip
20190830175710-172-31-29-220_precalc_no1_abcdefghij_10_3.zip
20190830175632-172-31-4-107_precalc_no5_abcdef_10_3.zip
20190830175734-172-31-4-107_precalc_no2_abcdef_10_3.zip
20190830180256-172-31-31-158_precalc_no3_abcdefghijklmno_20_3.zip
20190830175912-172-31-31-158_precalc_no3_abcdefghij_20_3.zip
20190830180016-172-31-31-158_precalc_no4_abcdefghij_20_3.zip
20190830175944-172-31-31-158_precalc_no2_abcdefghijklmno_10_3.zip
20190830175807-172-31-31-158_precalc_no4_abcdefghijklmno_15_3.zip
20190830180050-172-31-4-107_precalc_no1_abcdef_10_3.zip
20190830175912-172-31-4-107_precalc_no2_abcdefghijklmno_20_3.zip
20190830175851-172-31-29-220_precalc_no3_abcdefghij_10_3.zip
20190830175702-172-31-31-158_precalc_no1_abcdefghij_20_3.zip
20190830175945-172-31-4-107_precalc_no4_abcdefghijklmno_20_3.zip
20190830180120-172-31-31-158_precalc_no4_abcdef_10_3.zip
20190830180228-172-31-4-107_precalc_no4_abcdefghijklmno_10_3.zip
20190830175819-172-31-29-220_precalc_no2_abcdefghij_15_3.zip
20190830180116-172-31-29-220_precalc_no2_abcdefghij_10_3.zip
20190830175702-172-31-4-107_precalc_no5_abcdefghijklmno_20_3.zip
20190830180017-172-31-4-107_precalc_no2_abcdefghijklmno_15_3.zip
20190830180123-172-31-4-107_precalc_no3_abcdefghij_15_3.zip
20190830175734-172-31-31-158_precalc_no4_abcdefghij_10_3.zip
20190830175747-172-31-29-220_precalc_no2_abcdefghij_20_3.zip
20190830180048-172-31-31-158_precalc_no2_abcd_10_3.zip
20190830180006-172-31-29-220_precalc_no5_abcd_10_3.zip
20190830180303-172-31-29-220_precalc_no3_abcd_10_3.zip
20190830180224-172-31-31-158_precalc_no1_abcd_10_3.zip
20190830180038-172-31-29-220_precalc_no1_abcdefghijklmno_15_3.zip"""
#     precalcs = """20190528025718-172-31-27-212_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
# 20190528025828-172-31-30-230_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
# 20190528025939-172-31-17-144_precalc_no4_abcdefghijklmnopqrst_15_3.zip
# 20190528030045-172-31-18-173_precalc_no2_abcdefghijklmno_20_3.zip
# 20190528025940-172-31-31-24_precalc_no5_abcdefghij_10_3.zip
# 20190528030106-172-31-20-175_precalc_no5_abcdefghijklmno_20_3.zip
# 20190528025905-172-31-30-230_precalc_no1_abcdefghijklmnopqrst_20_3.zip
# 20190528030226-172-31-20-175_precalc_no5_abcdef_10_3.zip
# 20190528025904-172-31-17-144_precalc_no1_abcdefghij_15_3.zip
# 20190528030147-172-31-20-175_precalc_no3_abcdefghij_10_3.zip
# 20190528030014-172-31-17-144_precalc_no3_abcdefghijklmnopqrst_10_3.zip
# 20190528025748-172-31-18-129_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
# 20190528025940-172-31-30-230_precalc_no3_abcdefghijklmnopqrst_20_3.zip
# 20190528025904-172-31-31-24_precalc_no4_abcdefghijklmnopqrst_20_3.zip
# 20190528030016-172-31-31-24_precalc_no3_abcdefghij_20_3.zip
# 20190528030050-172-31-17-144_precalc_no4_abcdefghij_10_3.zip
# 20190528030229-172-31-18-173_precalc_no1_ab_10_3.zip
# 20190528025940-172-31-27-212_precalc_no4_abcdefghijklmno_20_3.zip
# 20190528025903-172-31-17-48_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
# 20190528030200-172-31-18-54_precalc_no2_abcdef_10_3.zip
# 20190528030050-172-31-31-24_precalc_no3_abcdefghijklmnopqrst_15_3.zip
# 20190528030050-172-31-27-212_precalc_no5_abcdefghijklmno_15_3.zip
# 20190528025935-172-31-18-129_precalc_no3_abcdefghijklmno_15_3.zip
# 20190528030125-172-31-17-144_precalc_no1_abcdefghijklmnopqrst_10_3.zip
# 20190528030200-172-31-27-212_precalc_no4_abcdef_10_3.zip
# 20190528030049-172-31-31-254_precalc_no4_abcdefghijklmnopqrst_10_3.zip
# 20190528025827-172-31-31-254_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
# 20190528030235-172-31-30-230_precalc_no3_ab_10_3.zip
# 20190528030050-172-31-17-48_precalc_no3_abcdefghijklmno_20_3.zip
# 20190528030120-172-31-18-173_precalc_no2_abcdefghijklmno_10_3.zip
# 20190528025718-172-31-30-230_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
# 20190528025903-172-31-18-54_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
# 20190528030015-172-31-31-254_precalc_no1_abcdefghijklmno_20_3.zip
# 20190528025828-172-31-18-54_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
# 20190528025718-172-31-31-24_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
# 20190528025717-172-31-17-48_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
# 20190528030120-172-31-18-129_precalc_no3_abcdefghijklmno_10_3.zip
# 20190528025829-172-31-20-175_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
# 20190528025939-172-31-17-48_precalc_no5_abcdefghijklmnopqrst_10_3.zip
# 20190528030159-172-31-31-254_precalc_no4_abcd_10_3.zip
# 20190528025752-172-31-31-254_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
# 20190528025640-172-31-27-212_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
# 20190528025713-172-31-18-129_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
# 20190528025641-172-31-17-48_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
# 20190528030015-172-31-18-54_precalc_no5_abcdefghij_15_3.zip
# 20190528030015-172-31-27-212_precalc_no5_abcdefghij_20_3.zip
# 20190528030230-172-31-18-129_precalc_no2_ab_10_3.zip
# 20190528025641-172-31-20-175_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
# 20190528025754-172-31-20-175_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
# 20190528030235-172-31-18-54_precalc_no5_ab_10_3.zip
# 20190528025640-172-31-18-173_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
# 20190528025934-172-31-18-173_precalc_no4_abcdefghij_15_3.zip
# 20190528025906-172-31-20-175_precalc_no1_abcdefghij_20_3.zip
# 20190528025640-172-31-31-254_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
# 20190528025747-172-31-18-173_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
# 20190528025753-172-31-31-24_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
# 20190528025753-172-31-17-144_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
# 20190528025857-172-31-18-173_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
# 20190528030125-172-31-31-254_precalc_no2_abcdefghij_10_3.zip
# 20190528030009-172-31-18-173_precalc_no5_abcdefghijklmno_10_3.zip
# 20190528025717-172-31-31-254_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
# 20190528030155-172-31-18-129_precalc_no1_abcd_10_3.zip
# 20190528030125-172-31-17-48_precalc_no2_abcdefghijklmnopqrst_15_3.zip
# 20190528030045-172-31-18-129_precalc_no5_abcdefghijklmnopqrst_20_3.zip
# 20190528030126-172-31-27-212_precalc_no4_abcdefghij_20_3.zip
# 20190528025828-172-31-31-24_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
# 20190528025717-172-31-17-144_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
# 20190528025712-172-31-18-173_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
# 20190528025753-172-31-30-230_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
# 20190528025940-172-31-18-54_precalc_no1_abcdefghij_10_3.zip
# 20190528030015-172-31-17-48_precalc_no4_abcdefghijklmno_15_3.zip
# 20190528030050-172-31-18-54_precalc_no4_abcdefghijklmno_10_3.zip
# 20190528025822-172-31-18-173_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCD_15_3.zip
# 20190528025905-172-31-27-212_precalc_no2_abcdefghij_15_3.zip
# 20190528025717-172-31-18-54_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
# 20190528030200-172-31-17-144_precalc_no5_abcd_10_3.zip
# 20190528030200-172-31-17-48_precalc_no3_abcd_10_3.zip
# 20190528025828-172-31-27-212_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
# 20190528030200-172-31-31-24_precalc_no3_abcdef_10_3.zip
# 20190528025827-172-31-17-144_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
# 20190528025640-172-31-30-230_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
# 20190528025858-172-31-18-129_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
# 20190528030200-172-31-30-230_precalc_no2_abcd_10_3.zip
# 20190528025753-172-31-17-48_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
# 20190528030126-172-31-18-54_precalc_no1_abcdefghijklmno_10_3.zip
# 20190528030126-172-31-30-230_precalc_no2_abcdefghij_20_3.zip
# 20190528030027-172-31-20-175_precalc_no2_abcdefghijklmno_15_3.zip
# 20190528025902-172-31-31-254_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_15_3.zip
# 20190528030016-172-31-30-230_precalc_no2_abcdefghijklmnopqrst_10_3.zip
# 20190528030235-172-31-27-212_precalc_no4_ab_10_3.zip
# 20190528030125-172-31-31-24_precalc_no1_abcdefghijklmnopqrst_15_3.zip
# 20190528025640-172-31-31-24_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCD_20_3.zip
# 20190528030010-172-31-18-129_precalc_no5_abcdefghijklmnopqrst_15_3.zip
# 20190528025640-172-31-17-144_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_15_3.zip
# 20190528025719-172-31-20-175_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
# 20190528025640-172-31-18-54_precalc_no3_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_10_3.zip
# 20190528025753-172-31-18-54_precalc_no2_abcdefghijklmnopqrtsuvwxyzABCD_10_3.zip
# 20190528025639-172-31-18-129_precalc_no4_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWX_20_3.zip
# 20190528025828-172-31-17-48_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
# 20190528025946-172-31-20-175_precalc_no3_abcdefghij_15_3.zip
# 20190528025823-172-31-18-129_precalc_no5_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_20_3.zip
# 20190528025753-172-31-27-212_precalc_no1_abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMN_10_3.zip
# 20190528030051-172-31-30-230_precalc_no1_abcdefghijklmno_15_3.zip
# 20190528030154-172-31-18-173_precalc_no1_abcdef_10_3.zip
# 20190528025939-172-31-31-254_precalc_no2_abcdefghijklmnopqrst_20_3.zip
# 20190822124012-172-31-21-193_precalc_4-5.zip
# 20190822124004-172-31-6-215_precalc_masaki_1-1.zip
# 20190822124005-172-31-8-47_precalc_2-2.zip
# 20190822124012-172-31-31-82_precalc_3-4.zip
# 20190830180152-172-31-31-158_precalc_no3_abcdefghijklmno_10_3.zip
# 20190830180328-172-31-31-158_precalc_no4_abcd_10_3.zip
# 20190830175929-172-31-29-220_precalc_no5_abcdefghij_15_3.zip
# 20190830175632-172-31-31-158_precalc_no5_abcdefghij_20_3.zip
# 20190830180335-172-31-29-220_precalc_no1_abcdefghijklmno_10_3.zip
# 20190830175634-172-31-29-220_precalc_no1_abcdefghij_15_3.zip
# 20190830175840-172-31-4-107_precalc_no1_abcdefghijklmno_20_3.zip
# 20190830175839-172-31-31-158_precalc_no5_abcdefghij_10_3.zip
# 20190830180155-172-31-4-107_precalc_no5_abcdefghijklmno_15_3.zip
# 20190830180230-172-31-29-220_precalc_no3_abcdefghijklmno_15_3.zip
# 20190830180154-172-31-29-220_precalc_no3_abcdef_10_3.zip
# 20190830175807-172-31-4-107_precalc_no5_abcdefghijklmno_10_3.zip
# 20190830180301-172-31-4-107_precalc_no4_abcdefghij_15_3.zip
# 20190830175710-172-31-29-220_precalc_no1_abcdefghij_10_3.zip
# 20190830175632-172-31-4-107_precalc_no5_abcdef_10_3.zip
# 20190830175734-172-31-4-107_precalc_no2_abcdef_10_3.zip
# 20190830180256-172-31-31-158_precalc_no3_abcdefghijklmno_20_3.zip
# 20190830175912-172-31-31-158_precalc_no3_abcdefghij_20_3.zip
# 20190830180016-172-31-31-158_precalc_no4_abcdefghij_20_3.zip
# 20190830175944-172-31-31-158_precalc_no2_abcdefghijklmno_10_3.zip
# 20190830175807-172-31-31-158_precalc_no4_abcdefghijklmno_15_3.zip
# 20190830180050-172-31-4-107_precalc_no1_abcdef_10_3.zip
# 20190830175912-172-31-4-107_precalc_no2_abcdefghijklmno_20_3.zip
# 20190830175851-172-31-29-220_precalc_no3_abcdefghij_10_3.zip
# 20190830175702-172-31-31-158_precalc_no1_abcdefghij_20_3.zip
# 20190830175945-172-31-4-107_precalc_no4_abcdefghijklmno_20_3.zip
# 20190830180120-172-31-31-158_precalc_no4_abcdef_10_3.zip
# 20190830180228-172-31-4-107_precalc_no4_abcdefghijklmno_10_3.zip
# 20190830175819-172-31-29-220_precalc_no2_abcdefghij_15_3.zip
# 20190830180116-172-31-29-220_precalc_no2_abcdefghij_10_3.zip
# 20190830175702-172-31-4-107_precalc_no5_abcdefghijklmno_20_3.zip
# 20190830180017-172-31-4-107_precalc_no2_abcdefghijklmno_15_3.zip
# 20190830180123-172-31-4-107_precalc_no3_abcdefghij_15_3.zip
# 20190830175734-172-31-31-158_precalc_no4_abcdefghij_10_3.zip
# 20190830175747-172-31-29-220_precalc_no2_abcdefghij_20_3.zip
# 20190830180048-172-31-31-158_precalc_no2_abcd_10_3.zip
# 20190830180006-172-31-29-220_precalc_no5_abcd_10_3.zip
# 20190830180303-172-31-29-220_precalc_no3_abcd_10_3.zip
# 20190830180224-172-31-31-158_precalc_no1_abcd_10_3.zip
# 20190830180038-172-31-29-220_precalc_no1_abcdefghijklmno_15_3.zip"""

    precalcs = make_list(precalcs)

    tasks = []
    for extracted in extracteds:
        splitted = extracted.split("_")[2:]
        tail = "_".join(splitted)
        s3 = f"eval_perf_{tail}"
        if "adfa" in extracted:
            precalc = "hoge"
        elif "paren" in extracted:
            precalc = "hoge"
        else:
            precalc = next(x for x in precalcs if tail in x)
        tasks.append({"s3": s3, "extracted": extracted, "precalc": precalc})
    return tasks


# Takes a task and return the result
def calc(d):
    z = eval_perf.run(d["extracted"], d["precalc"], d["s3"])
    # util.notify_slack(z + "/" + str(d))
    return z


# Called when no tasks are assgined to a worker
def handle_finish_machine(uri, name):
    pass


# Called when all the tasks are processed
def handle_finish_tasks():
    util.notify_slack("finished all!")
    pass


# Returns a string that is shown when 'http://(hostname):8080/' is accessed
def show_status():
    return "I'm working well!" + state_worker[0]


# ------ User write above ------

def get_hash(o):
    return hashlib.md5(pickle.dumps(o)).hexdigest()


# Prepare Flask
app = Flask(__name__)


# Make pages
@app.route("/", methods=["GET"])
def respond_home():
    return show_status()


def invoke_calc(task):
    global state_worker
    try:
        res = calc(copy.copy(task))
        lock_state.acquire()
        state_worker = ("done", task, res)
        lock_state.release()
    except Exception as e:
        import traceback, io
        with io.StringIO() as f:
            traceback.print_exc(file=f)
            app.logger.error(f.getvalue())
            lock_state.acquire()
            state_worker = ("error", task, f.getvalue())
            lock_state.release()


@app.route("/calc", methods=["POST"])
def respond_calc():
    global state_worker
    app.logger.info("Got calculation request".format())
    task = pickle.loads(request.data)
    try:
        lock_state.acquire()
        if state_worker[0] == "idle":
            state_worker = ("working", task)
            threading.Thread(target=invoke_calc, args=(task,)).start()
            app.logger.info("Accepting task".format())
        else:
            lock_state.release()
            app.logger.info("Rejecting the request because the state is {0}".format(state_worker[0]))
            abort(503, {})
        lock_state.release()
    except Exception as e:
        import traceback, io
        with io.StringIO() as f:
            traceback.print_exc(file=f)
            app.logger.error(f.getvalue())
            state_worker = ("error", task, f.getvalue())
    return "Accepted your task"


@app.route("/retrieve", methods=["POST"])
def respond_retrieve():
    global state_worker
    response = make_response()
    app.logger.info("Got retrieval request".format())
    task_request = pickle.loads(request.data)
    task_request_hash = get_hash(task_request)
    if state_worker[0] == "idle":
        app.logger.error("The state was idle".format())
        abort(404, {}) #404
    elif state_worker[0] == "working":
        app.logger.info("The state was working".format())
        abort(503, {})  # Service Unavailable
    elif state_worker[0] == "done":
        app.logger.info("The state was done".format())
        lock_state.acquire()
        _, task, res = state_worker
        task_hash = get_hash(task)
        if task_hash != task_request_hash:
            app.logger.error("The task we have done and the task of the request are different".format(),
                             extra={"who": "retrieve"})
            app.logger.error("Task we have: {0}".format(task),
                             extra={"who": "retrieve"})
            app.logger.error("Task of request: {0}".format(task_request),
                             extra={"who": "retrieve"})
            lock_state.release()
            abort(404, {})
        res = pickle.dumps({"task": task, "result": pickle.dumps(res)})
        response.data = res
        response.mimetype = "application/octet-stream"
        state_worker = ("idle",)
        app.logger.info("Returning the result".format())
        lock_state.release()
        return response
    elif state_worker[0] == "error":
        app.logger.info("The state was error".format())
        lock_state.acquire()
        _, task, error = state_worker
        res = pickle.dumps({"task": task, "error": error})
        response.data = res
        response.mimetype = "application/octet-stream"
        state_worker = ("idle",)
        lock_state.release()
        return response
    app.logger.info("Unexpected state {0}".format(state_worker))
    abort(500, {}) # Internal server error


@app.route("/status", methods=["GET"])
def respond_status():
    global state_worker
    return state_worker[0]


def caller(server, tasks, finisheds, lock, total):
    time_start = time.time()
    uri_server = server[0]
    name_server = server[1]
    rootLogger.info("Starting {0}@{1}".format(name_server, uri_server), extra={"who": name_server})
    while True:
        lock.acquire()
        if tasks == []:
            lock.release()
            break
        task = tasks.pop()
        rootLogger.info("Popped".format(), extra={"who": name_server})
        lock.release()
        try:
            filename = "{0}_{1}.done.pickle".format(name_server, get_time_hash())
            filename_error = "{0}_{1}.error.pickle".format(name_server, get_time_hash())
            data = pickle.dumps(task)
            res = requests.post(uri_server + "calc", data=data, timeout=timeout)
            if res.status_code == 200:
                rootLogger.info("Request is accepted".format(), extra={"who": name_server})
                while True:
                    time.sleep(interval_polling)
                    # rootLogger.info("Polling".format(), extra={"who": name_server})
                    res2 = requests.post(uri_server + "retrieve", data=data, timeout=timeout)
                    if res2.status_code == 200:
                        res2.raw.decode_content = True
                        res = pickle.loads(res2.content)
                        if "result" in res:
                            with open(filename, "wb") as f:
                                rootLogger.info("Saving the result as {0}".format(filename),
                                                extra={"who": name_server})
                                f.write(res2.content)
                            break
                        elif "error" in res:
                            rootLogger.info("Internal error occurred in the remote machine on the task: {0}".format(res["error"]),
                                            extra={"who": name_server})
                            with open(filename_error, "wb") as f:
                                rootLogger.info("Saving the error result as {0}".format(filename_error),
                                                extra={"who": name_server})
                                f.write(res2.content)
                            break
                        else:
                            raise Exception("Invalid result is given")
                    elif res2.status_code == 503:
                        pass  # the remote is working
                    elif res2.status_code == 404:
                        raise Exception("The remote machine is in idle.  The task was gone away...")
                    else:
                        raise Exception("Got unexpected error code {0}".format(res2.status_code))
                time_elapsed = time.time() - time_start
                lock.acquire()
                finisheds.append(task)
                lock.release()
                speed = time_elapsed / len(finisheds)
                eta = speed*(total - len(finisheds))
                rootLogger.info("Finished {0}/{1} tasks ({2} in the queue).  ETA is {3}".format(len(finisheds), total, len(tasks), eta),
                                extra={"who": name_server})
            else:
                rootLogger.info("Retrieving failed with {1}".format(res.status_code), extra={"who": name_server})
        except Exception as e:
            import traceback, io
            with io.StringIO() as f:
                traceback.print_exc(file=f)
                rootLogger.error("Request failed with the following error: {0}".format(f.getvalue()),
                                 extra={"who": name_server})
            # put the failed task back to the queue
            rootLogger.info("Putting back the failed task to the queue", extra={"who": name_server})
            lock.acquire()
            tasks.append(task)
            lock.release()
            break

    rootLogger.info("Closing".format(), extra={"who": name_server})
    handle_finish_machine(uri_server, name_server)


def get_servers():
    servers = [x.strip() for x in hosts.split("\n") if x.strip() != ""]
    servers = ["http://{0}:8080/".format(x) for x in servers]
    servers = [(server, name + str(i)) for i, server in enumerate(servers)]
    return servers

def get_status_remotes():
    xs = []
    servers = get_servers()
    for server in servers:
        try:
            res = requests.get(server[0] + "status")
            xs.append((server[0], res.content.decode()))
        except:
            xs.append((server[0], "error"))
    return xs

if __name__ == "__main__":
    # Prepare logger
    global rootLogger
    global logFile

    if len(sys.argv) < 2:
        print("No modes specified")
        sys.exit(1)

    if sys.argv[1] == "worker":
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    else:
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(who)s]  %(message)s")

    rootLogger = logging.getLogger("oden")
    rootLogger.setLevel(logging.INFO)

    logFile = "{0}.log".format(get_time_hash())
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Main
    # print(sys.argv[1])

    if sys.argv[1] in ["manager", "test", "resume"]:
        tasks = make_tasks()
        resume = False
        if sys.argv[1] == "resume":
            resume = True
            rootLogger.info("Starting resume mode".format(), extra={"who": "resume"})
            # hoge
            hash2task = {get_hash(v): v for v in tasks}
            tasks_hash = [get_hash(t) for t in tasks]
            tasks_mset = {h: tasks_hash.count(h) for h in hash2task.keys()}
            dones = []
            for i in pathlib.Path(".").glob("{0}*.done.pickle".format(name)):
                with open(i, "rb") as f:
                    dones.append(pickle.load(f)["task"])
            dones_hash = [get_hash(t) for t in dones]
            dones_mset = {h: dones_hash.count(h) for h in hash2task.keys()}
            remaining_mset = {h: tasks_mset[h] - dones_mset[h] for h in hash2task.keys()}
            tasks = []
            for k, v in remaining_mset.items():
                tasks += [copy.copy(hash2task[k]) for _ in range(v)]

            rootLogger.info("Loaded {0} tasks".format(len(tasks)), extra={"who": "resume"})
            sys.argv[1] = "manager"
        if sys.argv[1] == "manager":
            rootLogger.info("Starting manager mode", extra={"who": "manager"})
            # check if the remotes are ready
            for i in get_status_remotes():
                if i[1] != "idle":
                    rootLogger.error("Machine {0} is not in idle ({1}).  Cannot start the calculation.".format(i[0], i[1]))
                    sys.exit(1)
            #check if the previous calculation remaining
            if any(pathlib.Path(".").glob("{0}*.pickle".format(name))) and (not resume):
                ans = ""
                while True:
                    ans = input("The previous calculations seems remaining.  Do you really start the calculation? (y/n)")
                    if ans.lower().startswith("y"):
                        break
                    elif ans.lower().startswith("n"):
                        sys.exit(1)

            #
            servers = get_servers()
            rootLogger.info("Servers: " + str(servers), extra={"who": "manager"})

            lock = threading.Lock()
            num_tasks = len(tasks)
            finisheds = []
            rootLogger.info("We have {0} tasks.".format(num_tasks), extra={"who": "manager"})
            threads = []
            for server in servers:
                t = threading.Thread(target=caller, args=(server, tasks, finisheds, lock, num_tasks))
                t.start()
                threads.append(t)
            while True:
                if all([(not t.is_alive()) for t in threads]):
                    handle_finish_tasks()
                    break
        elif sys.argv[1] == "test":
            rootLogger.info("Starting test mode", extra={"who": "test"})
            with open("test.log", "w") as f:
                for i in make_tasks():
                    rootLogger.info("Starting task {0}".format(i), extra={"who": "manager"})
                    res = calc(i)
                    f.write(res + "\n")
    elif sys.argv[1] == "worker":
        if len(sys.argv) > 2:
            port = int(sys.argv[2])
        else:
            port = 8080
        app.run(host='0.0.0.0', port=port)
    elif sys.argv[1] == "status":
        for i in get_status_remotes():
            print("{0}\t{1}".format(i[0], i[1]))
    else:
        rootLogger.fatal("Invalid argument {0}".format(sys.argv[1]), extra={"who": "error"})


