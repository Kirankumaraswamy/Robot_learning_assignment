import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

training_accuracy_history0 = []
validation_accuracy_history0 = []
for e in tf.train.summary_iterator(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard/train/car_racing-20210608-113341/events.out.tfevents.1623144823.kiran-Inspiron-7591")):
    for v in e.summary.value:
        print(v.tag, v.simple_value)
        if v.tag == "episode_reward_1":
            training_accuracy_history0.append(v.simple_value)

plt.plot(training_accuracy_history0)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.title("Training reward for carracing")
plt.savefig("carracing.png")
plt.show()