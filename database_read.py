import os
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd


def graph_mobility():

    db_conn = sqlite3.connect("metropolis.db")
    db = db_conn.cursor()
    db.execute("SELECT * FROM simulations WHERE lattice_size=300")
    data_storage = "/Users/alisher/PycharmProjects/SpinGlass/data_storage"

    for row in db.fetchall():
        ext = row[3]
        if row[3] == 5 and row[4] == 119:
            g = np.load(os.path.join(data_storage, row[10]))
            sns.kdeplot(data=g, bw_adjust=2, label="T = {}".format(round(row[2], 3)))
            plt.xlim([0, 150])

    plt.legend()
    plt.title("Mobility under external field = {}".format(ext))
    plt.show()
    db_conn.commit()
    db_conn.close()


def graph_spin_distribution():

    db_conn = sqlite3.connect("metropolis.db")
    db = db_conn.cursor()
    db.execute("SELECT * FROM simulations WHERE lattice_size=300")
    data_storage = "/Users/alisher/PycharmProjects/SpinGlass/data_storage"

    for row in db.fetchall():
        if row[4] == 119:
            temp = row[2]
            g = np.load(os.path.join(data_storage, row[9]))
            # plt.plot(g, label="{}".format(row[2]))
            sns.kdeplot(data=g, bw_adjust=2, label="T = {}".format(round(row[3], 3)))
            plt.xlim([-1, 1])

    plt.legend()
    plt.title("Temperature = {}".format(temp))
    plt.show()
    db_conn.commit()
    db_conn.close()

    return None


def graph_approximate_energy():

    db_conn = sqlite3.connect("metropolis.db")
    db = db_conn.cursor()
    db.execute("SELECT * FROM simulations WHERE lattice_size=300")

    emf_temperatures = defaultdict(list)
    emf_energies = defaultdict(list)

    for row in db.fetchall():
        emf_temperatures[row[3]].append(row[2])
        emf_energies[row[3]].append(float(row[5][1:-1]))

    for emfs in emf_temperatures.keys():
        plt.scatter(emf_temperatures[emfs], emf_energies[emfs], label="External field = {}".format(emfs))

    plt.legend()
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.show()
    db_conn.commit()
    db_conn.close()


if __name__ == '__main__':

    database = "metropolis.db"
    conn = sqlite3.connect(database)
    df = pd.read_sql_query("SELECT * from simulations", conn)

    # print(df["seed"].median())
    # print(df.median())
    # # print(df["temperature"].value_counts(normalize=True))
    # df_emfs = df.groupby(["emf"])
    # print(df_emfs['temperature'].value_counts())
    # desired = df_emfs.get_group(0)

    cond1 = df['emf'] == 0
    cond2 = df['seed'] == 119

    print(df[cond1][cond2])

    # print(df.head())
    # # df.plot(x="temperature", y=["results_final_energy"  "emf" == 0])
    # # df.plot(kind='scatter', x='temperature', y='results_final_energy')
    # plt.show()
    # res = df.groupby("emf")["results_final_energy"]
    # print(res)
    # conn.close()
    graph_approximate_energy()
    graph_mobility()
    graph_spin_distribution()
