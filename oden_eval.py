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
import eval_rnn # ERASEHERE
import util


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


# Returns the list of the tasks
def make_tasks():
    extracteds = """20190827110441-172-31-28-104_rnn_no1_abcdefghijklmno_10_3.zip
20190827125600-172-31-27-60_rnn_no4_abcd_10_3.zip
20190827085359-172-31-27-42_rnn_no4_abcdefghij_15_3.zip
20190827125514-172-31-20-198_rnn_no3_abcd_10_3.zip
20190827090320-172-31-27-60_rnn_no3_abcdefghijklmno_20_3.zip
20190827090310-172-31-18-182_rnn_no3_abcdefghijklmno_15_3.zip
20190827070625-172-31-29-101_rnn_no4_abcdefghijklmno_10_3.zip
20190827130614-172-31-29-75_rnn_no1_abcd_10_3.zip
20190827070655-172-31-18-182_rnn_no5_abcdefghijklmno_15_3.zip
20190827090330-172-31-25-112_rnn_no3_abcdefghijklmno_10_3.zip
20190827125557-172-31-25-112_rnn_no3_abcdef_10_3.zip
20190827090530-172-31-20-198_rnn_no3_abcdefghij_15_3.zip
20190827125536-172-31-18-182_rnn_no4_abcdef_10_3.zip
20190827105954-172-31-27-60_rnn_no2_abcdefghij_10_3.zip
20190827130150-172-31-17-80_rnn_no1_abcdef_10_3.zip
20190827130301-172-31-28-104_rnn_no2_abcd_10_3.zip
20190827110040-172-31-20-198_rnn_no1_abcdefghijklmno_15_3.zip
20190827090911-172-31-17-80_rnn_no2_abcdefghijklmno_15_3.zip
20190827070707-172-31-25-112_rnn_no4_abcdefghij_20_3.zip
20190827125343-172-31-29-101_rnn_no5_abcd_10_3.zip
20190827070217-172-31-27-42_rnn_no4_abcdefghijklmno_20_3.zip
20190827104555-172-31-27-42_rnn_no2_abcdefghijklmno_10_3.zip
20190827071125-172-31-17-80_rnn_no5_abcdefghij_15_3.zip
20190827090650-172-31-23-42_rnn_no2_abcdefghijklmno_20_3.zip
20190827090609-172-31-28-104_rnn_no3_abcdefghij_20_3.zip
20190827090831-172-31-29-75_rnn_no3_abcdefghij_10_3.zip
20190827105956-172-31-25-112_rnn_no1_abcdefghijklmno_20_3.zip
20190827070825-172-31-28-104_rnn_no5_abcdefghij_10_3.zip
20190827105920-172-31-18-182_rnn_no2_abcdefghij_15_3.zip
20190827071031-172-31-23-42_rnn_no4_abcdefghijklmno_15_3.zip
20190827070909-172-31-20-198_rnn_no5_abcdefghijklmno_10_3.zip
20190827105821-172-31-29-101_rnn_no2_abcdefghij_20_3.zip
20190827090218-172-31-29-101_rnn_no4_abcdefghij_10_3.zip
20190827125741-172-31-23-42_rnn_no2_abcdef_10_3.zip
20190827110535-172-31-17-80_rnn_no1_abcdefghij_10_3.zip
20190827070645-172-31-27-60_rnn_no5_abcdefghijklmno_20_3.zip
20190827110218-172-31-23-42_rnn_no1_abcdefghij_20_3.zip
20190827123757-172-31-27-42_rnn_no5_abcdef_10_3.zip
20190827070915-172-31-29-75_rnn_no5_abcdefghij_20_3.zip
20190827110716-172-31-29-75_rnn_no1_abcdefghij_15_3.zip"""
    extracteds = extracteds.split("\n")
    extracteds = [x.strip() for x in extracteds]

    tasks = []
    for extracted in extracteds:
        splitted = extracted.split("_")[2:]
        tail = "_".join(splitted)
        s3 = f"posteval_{tail}"
        tasks.append({"s3": s3, "extracted": extracted})
    return tasks


# Takes a task and return the result
def calc(d):
    z = eval_rnn.run(args_s3rnn=d["extracted"],
                     args_s3wfa="",
                     args_batch_size=0,
                     args_max_length=20,
                     args_eps_relative=False,
                     args_eps=eval_rnn.default_eps,
                     args_s3=d["s3"],
                     args_skip_train=True,
                     args_embed_dim=0,
                     args_hidden_output_dims=[0])
    util.notify_slack(z + "/" + str(d))
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
        abort(404, {})  # 404
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
    abort(500, {})  # Internal server error


def caller(server, tasks, lock, total):
    time_start = time.time()
    uri_server = server[0]
    name_server = server[1]
    rootLogger.info("Starting {0}@{1}".format(name_server, uri_server), extra={"who": name_server})
    while True:
        lock.acquire()
        if tasks == []:
            break
        task = tasks.pop()
        rootLogger.info("Popped".format(), extra={"who": name_server})
        lock.release()
        try:
            filename = "{0}_{1}.pickle".format(name_server, get_time_hash())
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
                            rootLogger.info("Error occurred in the remote machine: {0}".format(res["error"]),
                                            extra={"who": name_server})
                            raise Exception("Error occurred in the remote machine")
                        else:
                            raise Exception("Invalid result is given")

                    elif res2.status_code == 503:
                        pass  # the remote is working
                    elif res2.status_code == 404:
                        raise Exception("The remote machine is in idle.  The task was gone away...")
                    else:
                        raise Exception("Got unexpected error code {0}".format(res2.status_code))
                time_elapsed = time.time() - time_start
                if total - len(tasks) == 0:
                    eta = 0
                else:
                    eta = time_elapsed * len(tasks) / (total - len(tasks))
                rootLogger.info("{0} tasks are remaining.  ETA is {1}".format(len(tasks), eta),
                                extra={"who": name_server})
            else:
                rootLogger.info("Retrieving failed with {1}".format(res.status_code), extra={"who": name_server})
        except Exception as e:
            import traceback, io
            with io.StringIO() as f:
                traceback.print_exc(file=f)
                rootLogger.error("Request failed with the following error: {0}".format(f.getvalue()),
                                 extra={"who": name_server})

    lock.release()
    rootLogger.info("Closing".format(), extra={"who": name_server})
    handle_finish_machine(uri_server, name_server)


if __name__ == "__main__":
    # Prepare logger
    global rootLogger

    if len(sys.argv) < 2:
        print("No modes specified")
        sys.exit(1)

    if sys.argv[1] == "worker":
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    else:
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(who)s]  %(message)s")

    rootLogger = logging.getLogger("oden")
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("{0}.log".format(get_time_hash()))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Main
    # print(sys.argv[1])

    if sys.argv[1] in ["manager", "test", "resume"]:
        tasks = make_tasks()
        if sys.argv[1] == "resume":
            rootLogger.info("Starting resume mode".format(), extra={"who": "resume"})
            # hoge
            hash2task = {get_hash(v): v for v in tasks}
            tasks_hash = [get_hash(t) for t in tasks]
            tasks_mset = {h: tasks_hash.count(h) for h in hash2task.keys()}
            dones = []
            for i in pathlib.Path(".").glob("{0}*.pickle".format(name)):
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
            servers = [x.strip() for x in hosts.split("\n") if x.strip() != ""]
            servers = ["http://{0}:{1}/".format(x, y) for x in servers for y in ports]
            servers = [(server, name + str(i)) for i, server in enumerate(servers)]
            rootLogger.info("Servers: " + str(servers), extra={"who": "manager"})

            lock = threading.Lock()
            num_tasks = len(tasks)
            rootLogger.info("We have {0} tasks.".format(num_tasks), extra={"who": "manager"})
            threads = []
            for server in servers:
                t = threading.Thread(target=caller, args=(server, tasks, lock, len(tasks)))
                t.start()
                threads.append(t)
            while True:
                if all([(not t.is_alive()) for t in threads]):
                    handle_finish_tasks()
                    break
        elif sys.argv[1] == "test":
            rootLogger.info("Starting test mode", extra={"who": "test"})
            for i in make_tasks():
                rootLogger.info("Starting task {0}".format(i), extra={"who": "manager"})
                calc(i)
                time.sleep(5)
    elif sys.argv[1] == "worker":
        if len(sys.argv) > 2:
            port = int(sys.argv[2])
        else:
            port = 8080
        app.run(host='0.0.0.0', port=port)
    else:
        rootLogger.fatal("Invalid argument {0}".format(sys.argv[1]), extra={"who": "error"})
