import sqlite3
from contextlib import closing, contextmanager

from prepare_dataset import read_hospital_data


@contextmanager
def open_demo_database():
    data = read_hospital_data()
    # Os nós síncronos do LangGraph podem rodar em threads diferentes.
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as database:
        database.row_factory = sqlite3.Row
        with database:
            database.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE patients (
                    id TEXT PRIMARY KEY,
                    encounter TEXT NOT NULL,
                    consent_recorded INTEGER NOT NULL CHECK (consent_recorded IN (0, 1))
                );
                CREATE TABLE exams (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients (id),
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'solicitado', 'em análise', 'liberado', 'cancelado'
                    ))
                );
                CREATE TABLE protocols (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL
                );
            """)
            database.executemany(
                "INSERT INTO patients VALUES (:id, :encounter, :consent_recorded)",
                data["patients"],
            )
            database.executemany(
                "INSERT INTO exams VALUES (:id, :patient_id, :name, :status)", data["exams"]
            )
            database.executemany(
                "INSERT INTO protocols VALUES (:id, :title, :text)", data["protocols"]
            )
        database.execute("PRAGMA query_only = ON")
        yield database


def consult_patient(database: sqlite3.Connection, patient_id: str) -> dict | None:
    patient = database.execute(
        "SELECT id, encounter, consent_recorded FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()
    if patient is None:
        return None

    exams = database.execute(
        "SELECT id, name, status FROM exams WHERE patient_id = ? ORDER BY id", (patient_id,)
    ).fetchall()
    protocols = database.execute("SELECT id, title, text FROM protocols ORDER BY id").fetchall()
    return {
        "patient": dict(patient),
        "exams": [dict(exam) for exam in exams],
        "protocols": {protocol["id"]: dict(protocol) for protocol in protocols},
    }
