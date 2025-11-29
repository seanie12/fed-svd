import os
import sys
import time

import atexit
import numpy as np
import torch.multiprocessing as mp

from misc.utils import *

def GPU_ID_CRITERION(worker_id, gpus_list):
    gpu_id = gpus_list[worker_id+1] if worker_id < len(gpus_list)-1 else gpus_list[(worker_id-(len(gpus_list)-1))%len(gpus_list)]
    return gpu_id

class ParentProcess:
    def __init__(self, args, Server, Client):
        self.args = args
        self.gpus = [int(g) for g in args.gpu.split(',')]
        self.gpu_server = self.gpus[0]
        self.proc_id = os.getppid()
        print(f'main process id: {self.proc_id}')

        self.worker_in_gpu = {key: [] for key in self.gpus}
        self.sd = mp.Manager().dict()
        self.sd['is_done'] = False
        self.create_workers(Client)
        self.server = Server(args, self.sd, self.gpu_server)
        atexit.register(self.done)

    def create_workers(self, Client):
        self.processes = []
        self.q = {}
        for worker_id in range(self.args.n_workers):
            gpu_id = GPU_ID_CRITERION(worker_id, self.gpus)
            self.worker_in_gpu[gpu_id].append(worker_id)
            print(f'worker_id: {worker_id}, gpu_id:{gpu_id}')
            self.q[worker_id] = mp.Queue()
            p = mp.Process(target=WorkerProcess, args=(self.args, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
            p.start()
            self.processes.append(p)

    def start(self):
        self.sd['is_done'] = False
        if not os.path.isdir(self.args.checkpt_path):
            os.makedirs(self.args.checkpt_path)
        if not os.path.isdir(self.args.log_path):
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.n_clients * self.args.frac)
        print('started!')

        for curr_rnd in range(self.args.n_rnds):
            if self.args.resume_from_round is not None and curr_rnd < self.args.resume_from_round:
                print(f'resume from checkpoint... skip round {curr_rnd}')
                continue

            self.curr_rnd = curr_rnd
            self.updated = set()
            np.random.seed(self.args.seed + curr_rnd)

            self.selected = []
            for gpu_id in self.gpus:
                selected_clients = list(np.random.permutation(self.worker_in_gpu[gpu_id])[:self.n_connected // len(self.gpus)])
                self.selected.extend(selected_clients)
            print(f'selected clients: {self.selected}')

            st = time.time()
            ##################################################
            self.server.on_round_begin(self.selected, curr_rnd)
            ##################################################
            while len(self.selected) > 0:
                _selected = []
                for worker_id, q in self.q.items():
                    c_id = self.selected.pop(0)
                    _selected.append(c_id)
                    q.put((c_id, curr_rnd), timeout=3)
                    if len(self.selected) == 0:
                        break
                    time.sleep(0.1) # Just enough to let the Queue finish
                self.wait(curr_rnd, _selected)
            ###########################################
            self.server.on_round_complete(self.updated, curr_rnd)
            ###########################################
            print(f'[main] round {curr_rnd} done ({time.time()-st:.2f} s)')

        self.sd['is_done'] = True
        for worker_id, q in self.q.items():
            q.put(None)
        print('[main] server done')
        sys.exit()

    def wait(self, curr_rnd, _selected):
        cont = True
        while cont:
            cont = False
            for c_id in _selected:
                if c_id not in self.sd:
                    cont = True
                else:
                    self.updated.add(c_id)
            time.sleep(1)

    def done(self):
        for p in self.processes:
            p.join()
        print('[main] All children have joined. Destroying main process ...')


class WorkerProcess:
    def __init__(self, args, worker_id, gpu_id, q, sd, Client):
        self.q = q
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.is_done = False

        self.client = Client(self.args, self.worker_id, self.gpu_id, self.sd)
        self.listen()

    def listen(self):
        while not self.sd['is_done']:
            mesg = self.q.get()
            if not mesg == None:
                client_id, curr_rnd = mesg
                ##################################
                self.client.switch_state(client_id, curr_rnd)
                self.client.on_receive_message(curr_rnd)
                self.client.on_round_begin(curr_rnd)
                self.client.save_state()
                ##################################
            time.sleep(1)

        print('[main] Terminating worker processes ... ')
        sys.exit()


