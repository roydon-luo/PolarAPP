import time

from polarapp.config import parse_train_config
from polarapp.trainer import run


def main():
    start_time = time.time()
    run(parse_train_config())
    print(f"Finished in {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
