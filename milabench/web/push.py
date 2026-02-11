import os
import traceback
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename

from milabench.metrics.archive import publish_zipped_run
from milabench.metrics.sqlalchemy import SQLAlchemy
from .utils import database_uri


def push_routes(app, database_uri):
    UPLOAD_FOLDER = '/tmp/'
    ALLOWED_EXTENSIONS = {'zip'}

    app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", UPLOAD_FOLDER)

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # We could also do a event push kind of thing
    @app.route('/api/push/zip', methods=['POST'])
    def upload_zip_file():
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/push')

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect('/push')

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(dest)

                with SQLAlchemy(database_uri, meta_override={}) as backend:
                    publish_zipped_run(backend, dest, stop_on_exception=True)

                os.remove(dest)
                return {
                    "status": "OK",
                    "message": f"{file.filename} was pushed"
                }
            except Exception as err:
                return {
                    "status": "ERR",
                    "message": f"{str(err)}"
                }

    @app.route('/api/push/folder/<string:jr_job_id>', methods=['GET'])
    def upload_job_folder(jr_job_id: str):
        """Push a job runner folder to the database"""

        from .slurm import safe_job_path
        from ..metrics.archive import publish_archived_run

        run_folder = safe_job_path(jr_job_id, "runs")
        failures = []
        success = []
        
        for run in os.scandir(run_folder):
            try:
                run_path = os.path.join(run_folder, run)

                with SQLAlchemy(database_uri, meta_override={}) as backend:
                    publish_archived_run(backend, run_path, stop_on_exception=True)

                success.append(run.name)

            except Exception as err:
                traceback.print_exc()
                failures.append((run.name, str(err)))

        print("DONE")
        return {
            "status": "OK",
            "success": success,
            "failures": failures,
        }



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


def push_server(config):
    """Simple push server that takes a zip folder of runs to push to the database"""

    DATABASE_URI = database_uri()

    app = Flask(__name__)
    app.config.update(config)

    push_routes(app, DATABASE_URI)

    return app


def main():
    # flask --app milabench.web.push:main run

    # curl -X POST -F "file=@your_file.zip" http://localhost:5000/push

    app = push_server({})
    return app


if __name__ == "__main__":
    main()
