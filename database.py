import logging
import datetime as dt
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement, dict_factory

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

POINTS = ['127.0.0.1']
PORT = 9042
KEYSPACE = 'mnist'



def create_keyspace ():
    cluster = Cluster(contact_points=POINTS, port=PORT)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = {'class':'SimpleStrategy', 'replication_factor':'2'}
            """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
            CREATE TABLE posts (
                post_id int,
                timekey timestamp,
                model text,
                result float,
                label text,
                PRIMARY KEY (post_id)
            )
            """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)


def query_all ():
    cluster = Cluster(contact_points=POINTS, port=PORT)
    session = cluster.connect(KEYSPACE)
    session.row_factory = dict_factory
    rows = session.execute("SELECT * FROM posts")
    
    D = dict()
    for row in rows:
        D[row['post_id']] = row

    return D


def insert_row (row):
    cluster = Cluster(contact_points=POINTS, port=PORT)
    session = cluster.connect(KEYSPACE)
    session.execute(
        """
        INSERT INTO posts (post_id, timekey, model, result, label)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (row['post_id'], dt.datetime.now(), row['model'], 
         row['result'], row['label'])
    )




if __name__ == '__main__':
    row = {'post_id':1, 'model':'mnist', 'result':1, 'label':'One'}
