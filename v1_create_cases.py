import queue
import random
import os.path
import threading

from ansys.mapdl.core import launch_mapdl
from ThesisProject.Cases import CaseV1
from ThesisProject.util_functions import get_random_token

N_START = 0
N = 50_000
N_THREADS = 16
N_PROCS = 1
RESOLUTION = 10
CASE_OUTPUT_DIR = 'output/cases_v1'
if not os.path.exists(CASE_OUTPUT_DIR):
    os.makedirs(CASE_OUTPUT_DIR)


def test():
    files = os.listdir((CASE_OUTPUT_DIR))
    file = files[0]

    case = CaseV1.from_file(os.path.join(CASE_OUTPUT_DIR, file))
    case.plot_field_matrices()


def worker(q):
    apdl_instance = launch_mapdl(override=True, nproc=N_PROCS, remove_temp_dir_on_exit=True)
    done_cases = os.listdir(CASE_OUTPUT_DIR)

    while True:
        case_n = q.get()
        if case_n is None:
            break

        file_name = f'{case_n:06}.bin'
        if file_name in done_cases:
            continue

        save_path = os.path.join(CASE_OUTPUT_DIR, file_name)

        case = CaseV1.random(case_n, RESOLUTION)
        case.analyze(apdl_instance)

        case.save(save_path)
        print(f'[done] case: {case_n}')

        q.task_done()

    apdl_instance.exit()
    print('Thread finished')


def main():
    q = queue.Queue()

    for case_n in range(N_START, N_START + N):
        q.put(case_n)

    threads = []
    for _ in range(N_THREADS):
        q.put(None)
        t = threading.Thread(target=worker, args=(q,))
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
    # test()
