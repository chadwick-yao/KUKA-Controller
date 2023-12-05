import multiprocessing as mp
import time


class SecTimeCounter(mp.Process):
    def __init__(self, limited_sec_time):
        super().__init__()

        self.limited_sec_time = limited_sec_time
        self.elapsed_time = 0

        self.start_event = mp.Event()
        self.stop_event = mp.Event()
        self.stop_record_event = mp.Event()

    def start(self):
        self.start_event.set()
        super().start()

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        start_time = time.time()
        while not self.stop_event.is_set():
            self.elapsed_time = time.time() - start_time
            if (
                self.elapsed_time >= self.limited_sec_time
                and not self.stop_record_event.is_set()
            ):
                self.stop_record_event.set()


if __name__ == "__main__":
    with SecTimeCounter(5) as timer:
        while True:
            if timer.stop_record_event.is_set():
                break
    # for _ in range(3):
    #     timer = SecTimeCounter(5)
    #     timer.start()
    #     time.sleep(7)
    #     timer.stop()
    #     time.sleep(3)
