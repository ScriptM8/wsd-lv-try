import psycopg2

# Connect to your postgres DB
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="rrr22678max123",
    host="localhost",  # replace with your host if different
    port="5432"  # replace with your port if different
)
cur = conn.cursor()


def return_all():
    # Execute SQL command
    cur.execute("""
        select a.heading, b.id as sense, b.gloss, c.content from dict.entries a, dict.senses b, dict.examples c
             where a.id = b.entry_id
             and b.id = c.sense_id;
    """)
    return cur.fetchall()
