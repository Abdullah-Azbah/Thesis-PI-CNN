import os
import queue
import threading

from ThesisProject.Cases import CaseV1

RESOLUTION = 100
BASE_CASE_DIR = f'output/cases_v1'
MATRICES_OUTPUT_DIR = f'output/cases_v1_res{RESOLUTION}'
N_THREADS = 32

if not os.path.exists(MATRICES_OUTPUT_DIR):
    os.makedirs(MATRICES_OUTPUT_DIR)

q = queue.Queue()

done_cases = os.listdir(MATRICES_OUTPUT_DIR)

for file in os.listdir(BASE_CASE_DIR):
    if file in done_cases:
        continue
    q.put(file)


def worker():
    while True:
        file = q.get()

        if file is None:
            q.task_done()
            break
        case = CaseV1.from_file(os.path.join(BASE_CASE_DIR, file))

        matrices = case.create_matrices(RESOLUTION)
        matrices.save(os.path.join(MATRICES_OUTPUT_DIR, file))

        print(file)
        q.task_done()


threads = []
for _ in range(N_THREADS):
    q.put(None)

    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
