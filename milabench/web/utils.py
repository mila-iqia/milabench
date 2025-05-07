
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


def plot(chart):
    return f"""
    <div>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        <div id="vis"></div>
        <script type="text/javascript">
            (function () {{
                const spec = {chart};
                vegaEmbed('#vis', spec, {{actions: false}}).catch(console.error);
            }})();
        </script>
    </div>
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


def make_selection_key(key, names=None):
    from milabench.metrics.sqlalchemy import Exec, Metric, Pack
    from sqlalchemy import Text, cast

    table, path = key.split(":")
    tables = {
        "Exec": Exec, 
        "Metric": Metric, 
        "Pack": Pack
    }

    types = {
        "product": str
    }

    maybe = path.split(" as ")
    path = maybe[0]

    frags = path.split(".")
    selection = getattr(tables[table], frags[0])

    for frag in frags[1:-1]:
        selection = selection[frag]

    if len(frags) > 1:
        lst = frags[-1]
        lst_type = types.get(lst, None)

        selection = selection[lst]

        if lst_type is not None:
            if lst_type is str:
                selection = cast(selection, Text)
    
    if len(maybe) == 2:
        as_name = maybe[1]
    else:
        as_name = key

    if names is not None:
        names[key] = as_name

    return selection.label(as_name)

def make_filter(key):
    op = key["operator"]
    field = make_selection_key(key["field"])
    value = key["value"]

    match op:
        case "in":
            return field.in_(value.split(","))
        case "not in":
            return field.notin_(value.split(","))
        case "==":
            return field == value
        case "!=":
            return field != value
        case ">":
            return field > value
        case "<":
            return field < value
        case ">=":
            return field >= value
        case "<=":
            return field <= value
        case "like":
            return field.like(value)
        case "not like":
            return field.notlike(value)

def make_filters(filters):
    return [make_filter(f) for f in filters]
