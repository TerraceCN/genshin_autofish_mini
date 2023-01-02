# -*- coding:utf-8 -*-
from argparse import ArgumentParser
import time

import keyboard
import winsound
from onnxruntime import InferenceSession
import numpy as np

from utils import Fishing


def make_parser():
    parser = ArgumentParser("Genshin AutoFish Mini")
    parser.add_argument(
        "--model_path", default="./weights/fish_genshin_net.onnx", type=str
    )

    return parser


def main(args: ArgumentParser):
    session = InferenceSession(args.model_path)

    print("Model loaded")
    while True:
        print('Press "R" to start fishing')
        winsound.Beep(500, 500)
        keyboard.wait("r")
        winsound.Beep(500, 500)
        start_fishing(session)


def start_fishing(session: InferenceSession):
    env = Fishing(delay=0.1, max_step=10000, show_det=False)

    print("Start fishing", end="")

    winsound.Beep(700, 500)
    while not env.is_bite():
        time.sleep(0.5)
        print(".", end="")

    print("Bited!")

    winsound.Beep(900, 500)
    env.drag()
    time.sleep(1)

    print("Drag fish", end="")

    state = env.reset()
    for i in range(env.max_step):
        state = np.array(state).astype(np.float32)
        action = session.run(None, {"input": state[np.newaxis, :]})
        action = np.argmax(action[0][0])
        state, reward, done = env.step(action)
        if i % 5 == 0:
            print(".", end="")
        if done:
            print("Done")
            break
    time.sleep(1)
    

if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
