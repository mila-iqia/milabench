
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
            </head>
            <body>
                <div class="container">
                    {body}
                </div>
            </body>
        </html>
        """
