import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import pickle
import datetime
import sys
import os
from proto import sim_debug_pb2
import tkinter.filedialog


class sim_data_recorder:
    def __init__(self, name: str = "sim_data"):
        self.name = name
        self.proto = sim_debug_pb2
        # self.fig, self.ax = plt.subplots()

    def set_file_prefix(self, prefix: str):
        self.name = prefix

    def get_file_name(self):
        prefix = self.name
        now_dt = datetime.datetime.now()
        dt_str = now_dt.strftime("%Y-%m-%d_%H-%M-%S")
        # suffix = re.sub([" ", ":"], "_", dt_str)
        return f"{prefix}_{dt_str}.pkl"

    def save_data(self, data):
        file_name = self.get_file_name()
        with open(file_name, "wb") as f:
            pickle.dump(data, f)

    def load_data(self):
        file_name = self.get_file_name()
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        return data


class sim_data_player:
    def __init__(self, name: str = "sim_data"):
        self.name = name
        self.proto = sim_debug_pb2
        self.recorder = sim_data_recorder()
        self.loaded_object = None

    def analyze_data(self):
        
        file_name = tkinter.filedialog.askopenfilename(
            title="select replay data file", filetypes=[("Pickle Files", "*.pkl")]
        )
    
        # Handle case where user cancels the dialog
        if not file_name:
            print("No file selected.")
            return
    
        try:
            with open(file_name, "rb") as file:
                self.loaded_object = pickle.load(file)
        except Exception as e:
            print(f"Failed to load file: {e}")
            return
    
        #####################################
        #########  Show Info History ########
        #####################################
        fig2, axes = plt.subplots(5, 1)
        fig2.canvas.manager.set_window_title("Ego Vehicle Motion Info")
    
        Hist_Sts = self.loaded_object.vehicle_state_debug
        Hist_Cmd = self.loaded_object.controller_debug
        time = self.loaded_object.times
    
        kappa = [item.kappa for item in Hist_Sts]
        axes[0].plot(time, kappa, c="r", label="kappa")
        axes[0].set_ylabel("$\kappa$")
        axes[0].grid(True)
    
        velocity = [item.velocity for item in Hist_Sts]
        axes[1].plot(time, velocity, c="b", label="velocity")
        axes[1].set_ylabel("$v$")
        axes[1].grid(True)
    
        acceleration_sts = [item.acceleration for item in Hist_Sts]
        axes[2].plot(time, acceleration_sts, c="g", label="acceleration")
        axes[2].set_ylabel("ax")
        axes[2].grid(True)
    
        kappa_rate = [item.kappa_rate for item in Hist_Cmd]
        axes[3].plot(time, kappa_rate, c="y", label="kappa_rate")
        axes[3].set_ylabel("kappa_rate")
        axes[3].grid(True)
    
        acceleration_cmd = [item.acceleration for item in Hist_Cmd]
        axes[4].plot(time, acceleration_cmd, c="b", label="acceleration")
        axes[4].set_ylabel("a")
        axes[4].grid(True)
    
        plt.show()


if __name__ == "__main__":
    player = sim_data_player()
    player.analyze_data()
