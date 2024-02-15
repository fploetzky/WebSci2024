import traceback
from ctypes import c_bool
import inspect
import sys
import os
import time


from multiprocess.queues import Empty
import numpy as np
from accelerate import Accelerator
import torch.multiprocessing as tmp
import multiprocess as mp

from tqdm import tqdm

CHANGE_DONE = "DONE"

class InferenceProcess(mp.Process):

    def __init__(self, prepare_model_and_tokenizer, on_batch_received, qi, qo):
        super().__init__(daemon=True)
        
        self.pmat = prepare_model_and_tokenizer
        self.obr = on_batch_received
        
        # qi: send from main, qo: send from subprocesses
        self.qi = qi
        self.qo = qo
        
        # pi: send from main, po: send from subprocess
        self.pi, self.po = mp.Pipe()

        self._do_listen = mp.Value(c_bool)
        self._do_listen.value = True

        self.start()
        
    def set_on_batch_received(self, on_batch_received):
        # find out, whether on_bon_batch_received depends on any state outside its scope
        global_vars =  inspect.getclosurevars(on_batch_received).globals
        if self.is_alive():
            #self.obr = on_batch_received
            self.pi.send((on_batch_received, global_vars))
            # wait for answer from process
            while not self.pi.poll():
                time.sleep(0.1)
            e = self.pi.recv()
            if e != CHANGE_DONE:
                raise e
        

    def stop(self):
        self._do_listen.value = False
        self.pi.close()
        self.po.close()
        self.join()
        
    def _run(self):
        import time
        state = Accelerator()
        
        print(state.process_index, "starting", state.device)

        start = time.time()
        
        model, tokenizer = self.pmat(state)

        diff = time.time() - start
        print(state.process_index, f"Loading model took: {diff:.2f}s")
        
        # wait for all processes to finish loading
        state.wait_for_everyone()

        while self._do_listen.value:
            # check for batch
                # if batch found: run batch
            # check for instruction in pipe
            
            try:
                i, batch = self.qi.get(block=False, timeout=0.1)
               
                #print(state.process_index, "got", i, batch)

                # run inference
                result = self.obr(batch, model, tokenizer)
                # send back result
                self.qo.put((i, result))

            except Empty:
                pass
            except Exception as e:
                self.po.send(e)
            
            try:
                if self.po.poll():
                    obr, global_vars = self.po.recv()
                    print(state.process_index, "got new instruction:", obr, global_vars)
                    self.obr = obr
                    # update globals
                    module = sys.modules["__main__"]
                    for name, value in global_vars.items():
                        setattr(module, name, value)
                    
                    # notify main about change done
                    self.po.send(CHANGE_DONE)

            except Exception as e:
                self.po.send(e)

        print(state.process_index, "terminating")
        
    def run(self):
        try:
            self._run()
        except Exception as e:
            self.po.send(e)

    def get_error(self):
        if not self.pi.closed and self.pi.poll():
            e = self.pi.recv()
            raise e

from accelerate.launchers import PrepareForLaunch, patch_environment
import torch

class InferenceContext():
    def __init__(self, num_gpus=9, verbose=0):
        self.num_gpus = num_gpus
        self.verbose = verbose
        # manager state
        self.is_initialized = False

        # state of current initialization
        # process context of last started processes
        self.processes = None

        # in and out queues
        self.qi, self.qo, = mp.Queue(), mp.Queue()

    def set_on_batch_received(self, on_batch_received):
        if not self.is_initialized:
            raise ValueError("Need to call start() first")
        
        if self.verbose:
            # check dependencies
            global_vars =  inspect.getclosurevars(on_batch_received).globals
            if len(global_vars) > 0:
                print(f"Your `on_batch_received` method relies on global variables. Their current state will be copied into the new processes,"
                      f" but, future changes will not be propagated automatically (yet).\n"
                      f"Here is the list of global variables: {global_vars}")
        for p in self.processes:
            p.set_on_batch_received(on_batch_received)

    def start(self, load_model_and_tokenizer, on_new_batch, num_processes=2, mixed_precision='no', use_port='29500', 
              master_addr='127.0.0.1', node_rank=0, num_nodes=1):

        if self.is_initialized:
            raise ValueError("I am already running! Call stop first.")
                
        with patch_environment(nproc=num_processes, node_rank=node_rank, world_size=num_nodes * num_processes,
                               master_addr=master_addr, master_port=use_port, mixed_precision=mixed_precision,):
            #launcher = PrepareForLaunch(inference_from_queue, distributed_type="MULTI_GPU")
            self.processes = []
            for index in range(num_processes):
                # accelerate requires some environment vars to be set to function properly
                # here, we choose a somewhat lazy approach to set the env vars for the spawned processes:
                # on each fork, the env vars of this process (main) are copied.
                # thus, we set the desired env vars before spawning a process
                os.environ["LOCAL_RANK"] = str(index)
                os.environ["RANK"] = str(num_processes * node_rank + index)
                os.environ["FORK_LAUNCHED"] = str(1)
                p = InferenceProcess(load_model_and_tokenizer, on_new_batch, self.qi, self.qo)
                self.processes.append(p)
            
        self.is_initialized = True

        
    def run_inference(self, data, max_batch_size=1):
        if not self.is_initialized:
            raise ValueError("Need to run start first")
        
        if isinstance(data, str):
            data = [data]
            batch_size=1
            
        # batching the data
        num_batches = int(np.ceil(len(data) / max_batch_size))
        
        # create progress bar
        try:
            with tqdm(total=num_batches) as pbar:

                results = []
                for i in range(num_batches):
                    batch = data[i*max_batch_size:(i+1)*max_batch_size]
                    self.qi.put((i, batch))

                    # try to read from self.qo until its empty
                    while not self.qo.empty():
                        results.append(self.qo.get())
                        pbar.update(1)

                # read until all batches are returned
                while len(results) != num_batches:
                    results.append(self.qo.get())
                    pbar.update(1)

        except KeyboardInterrupt:
            # clear up queues
            # issue: if we just submitted a long lasting batch to a subprocess
            # we can not make sure that subprocesses are still running on them
            while not self.qi.empty():
                self.qi.get()
            while not self.qo.empty():
                self.qo.get()
            return
                
        # sort results, in case batches got mixed up
        results = sorted(results, key=lambda x: x[0])
        # remove batch index, concat to list
        results = sum((x[1] for x in results), [])
        return results

    def stop(self):
        if self.is_initialized:
            
            # for future:
            '''
            for c1, c2 in self.pipes:
                c1.close()
                c2.close()
            self.qi.close()
            self.qo.close()
            '''
            for p in self.processes:
                p._do_listen.value = False
            for p in self.processes:
                p.stop()
            self.is_initialized = False
    
    def status(self):
        print("Initialized?", self.is_initialized)
        if self.is_initialized:
            print(self.processes)
