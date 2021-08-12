import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

training_accuracy_history0 = []
validation_accuracy_history0 = []
for e in tf.train.summary_iterator(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard/bc_history0-20210608-021737/events.out.tfevents.1623111462.kiran-Inspiron-7591")):
    for v in e.summary.value:
        if v.tag == "training_accuracy_1":
            training_accuracy_history0.append(v.simple_value)
        if v.tag == "validation_accuracy_1":
            validation_accuracy_history0.append(v.simple_value)


training_accuracy_history1 = []
validation_accuracy_history1 = []
for e in tf.train.summary_iterator(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard/bc_history1-20210608-022633/events.out.tfevents.1623111997.kiran-Inspiron-7591")):
    for v in e.summary.value:
        if v.tag == "training_accuracy_1":
            training_accuracy_history1.append(v.simple_value)
        if v.tag == "validation_accuracy_1":
            validation_accuracy_history1.append(v.simple_value)

training_accuracy_history3 = []
validation_accuracy_history3 = []
for e in tf.train.summary_iterator(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard/bc_history3-20210608-023417/events.out.tfevents.1623112463.kiran-Inspiron-7591")):
    for v in e.summary.value:
        if v.tag == "training_accuracy_1":
            training_accuracy_history3.append(v.simple_value)
        if v.tag == "validation_accuracy_1":
            validation_accuracy_history3.append(v.simple_value)

training_accuracy_history5 = []
validation_accuracy_history5 = []
for e in tf.train.summary_iterator(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard/bc_history5-20210608-024557/events.out.tfevents.1623113162.kiran-Inspiron-7591")):
    for v in e.summary.value:
        if v.tag == "training_accuracy_1":
            training_accuracy_history5.append(v.simple_value)
        if v.tag == "validation_accuracy_1":
            validation_accuracy_history5.append(v.simple_value)

fig, axs = plt.subplots(2, figsize=(15, 15))
fig.suptitle("Performance for Behaviour cloning with different history")
fig.tight_layout(pad=3.0)
axs[0].plot(np.arange(1, len(training_accuracy_history0) * 5, 5), training_accuracy_history0, label="history0")
axs[0].plot(np.arange(1, len(training_accuracy_history1) * 5, 5), training_accuracy_history1, label="history1")
axs[0].plot(np.arange(1, len(training_accuracy_history3) * 5, 5), training_accuracy_history3, label="history3")
axs[0].plot(np.arange(1, len(training_accuracy_history5) * 5, 5), training_accuracy_history5, label="history5")
axs[0].set(xlabel="Epoch", ylabel="Accuracy")
axs[0].set_title("Training performance")
axs[1].plot(np.arange(1, len(validation_accuracy_history0) * 5,5), validation_accuracy_history0, label="history0")
axs[1].plot(np.arange(1, len(validation_accuracy_history1) * 5,5), validation_accuracy_history1, label="history1")
axs[1].plot(np.arange(1, len(validation_accuracy_history3) * 5,5), validation_accuracy_history3, label="history3")
axs[1].plot(np.arange(1, len(validation_accuracy_history5) * 5,5), validation_accuracy_history5, label="history5")
axs[1].set(xlabel="Epoch", ylabel="Accuracy")
axs[1].set_title("Validation performance")
plt.legend()
plt.savefig("bc_cloning_history0.png")
plt.show()
