import os
import sqlite3
import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def graph_data():
    db_conn = sqlite3.connect("metropolis.db")
    db = db_conn.cursor()
    print( [float(i) for i in np.linspace(0, 7, 20)])
    db.execute("SELECT * FROM simulations WHERE lattice_size=40 AND emf = 5.526315789473684")
    # print(db.fetchall())
    # temperatures = []
    # energies = []
    # emf = []
    # for row in db.fetchall():
    #     temperatures.append(row[2])
    #     energies.append(float(row[5][1:-1]))
    #     emf.append(row[3])
    #
    # plt.scatter(emf, energies)
    # plt.show()
    data_storage = "/Users/alisher/PycharmProjects/SpinGlass/data_storage"
    for row in db.fetchall():
        g = np.load(os.path.join(data_storage, row[10]))
        #plt.plot(g, label="{}".format(row[2]))
        sns.kdeplot(data=g, bw_adjust=.5)
    plt.legend()
    plt.show()

    db_conn.commit()
    db_conn.close()

graph_data()