import sqlite3
""" Managing results of parametric sweep with sqlite3"""


def create_database() -> None:

    db_conn = sqlite3.connect("metropolis.db")
    db = db_conn.cursor()

    db.execute("""CREATE TABLE simulations (
                lattice_size       REAL,
                iterations         REAL,
                temperature        REAL,
                emf                REAL,
                seed               INTEGER,
                
                results_final_energy     TEXT,
                
                result_magnetism_array   TEXT,
                result_energy_array      TEXT,
                result_correlation_array TEXT,
                result_final_spin_array  TEXT,
                results_mobility_array   TEXT
                )
                """)
    db_conn.commit()
    db.close()
    return None


if __name__ == '__main__':
    create_database()
