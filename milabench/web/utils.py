
import os


def database_uri():
    USER = os.getenv("POSTGRES_USER", "username")
    PSWD = os.getenv("POSTGRES_PSWD", "password")
    DB = os.getenv("POSTGRES_DB", "milabench")
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = os.getenv("POSTGRES_PORT", 5432)

    uri_override = os.getenv("DATABASE_URI", None)

    return uri_override or f"postgresql://{USER}:{PSWD}@{HOST}:{PORT}/{DB}"


def page(title, body):
    css = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">'
    
    return f"""
        <!doctype html>
        <html>
            <head>
                <title>{title}</title>
                {css}

                <style>
                    th {{
                        text-align: left
                    }}

                    td {{
                        text-align: right
                    }}
                </style>
            </head>
            <body>
                <div class="container-fluid">
                    {body}
                </div>
            </body>
        </html>
        """



def cursor_to_json(cursor):
    columns = list(cursor.keys())
    results = []
    for row in cursor:
        row_dict = {}
        for col, val in zip(columns, row):
            row_dict[col] = val
        results.append(row_dict)
    return results


def cursor_to_dataframe(cursor):
    import pandas as pd

    columns = list(cursor.keys())
    results = []
    for row in cursor:
        row = list(row)
        results.append(row)

    return pd.DataFrame(results, columns=columns)


def make_selection_key(key):
    from milabench.metrics.sqlalchemy import Exec, Metric, Pack

    table, path = key.split(":")
    tables = {
        "Exec": Exec, 
        "Metric": Metric, 
        "Pack": Pack
    }

    maybe = path.split(" as ")
    path = maybe[0]

    frags = path.split(".")
    selection = getattr(tables[table], frags[0])

    for frag in frags[1:]:
        selection = selection[frag]

    if len(maybe) == 2:
        as_name = maybe[1]
    else:
        as_name = "_".join(frags)

    return selection.label(as_name)

def make_filter(key):
    pass