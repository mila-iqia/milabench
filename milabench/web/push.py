import os
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
import html

from milabench.metrics.archive import publish_zipped_run
from milabench.metrics.sqlalchemy import SQLAlchemy
from .utils import database_uri, page


def push_server(config):
    """Simple push server that takes a zip folder of runs to push to the database"""

    UPLOAD_FOLDER = '/tmp/'
    ALLOWED_EXTENSIONS = {'zip'}
    DATABASE_URI = database_uri()

    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", UPLOAD_FOLDER)
    app.config.update(config)

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    # We could also do a event push kind of thing

    @app.route('/push', methods=['GET', 'POST'])
    def upload_file():
        parts = []

        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['file']

            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(dest)

                    with SQLAlchemy(DATABASE_URI, meta_override={}) as backend:
                        publish_zipped_run(backend, dest, stop_on_exception=True)

                    os.remove(dest)
                    parts.append(f'<div class="alert alert-success" role="alert">{html.escape(file.filename)} was pushed</div>')
                except Exception as err:
                    parts.append(f'<div class="alert alert-danger" role="alert">{html.escape(str(err))}</div>')

        parts = "".join(parts)

        body = f"""
            <h1>Upload new File</h1>
            {parts}
            <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
        """
    
        return page("Upload run folder", body)
        
    return app


def push_zip_folder(file_path, url='http://localhost:5000/push'):
    #
    # TODO: zip the folder with python and upload it with requests
    #
    import requests

    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'application/zip')}
        response = requests.post(url, files=files)

    print(f"Status Code: {response.status_code}")
    print(response.text)


def main():
    # flask --app milabench.web.push:main run

    # curl -X POST -F "file=@your_file.zip" http://localhost:5000/push

    app = push_server({})
    return app


if __name__ == "__main__":
    main()
