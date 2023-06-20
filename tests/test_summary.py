from milabench.cli import main


def test_report(runs_folder, capsys, file_regression, config):
    folder = runs_folder / "rijubigo.2023-03-24_13:45:27.512446"
    try:
        main(["report", "--runs", folder, "--config", config("benchio")])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(folder), "XXX")
    file_regression.check(output)


def test_report_folder_does_average(runs_folder, capsys, file_regression):
    try:
        main(["report", "--runs", runs_folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(runs_folder), "XXX")
    file_regression.check(output)


def test_compare(runs_folder, capsys, file_regression):
    try:
        main(["compare", runs_folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(runs_folder), "XXX")
    file_regression.check(output)


def test_summary_full(runs_folder):
    from milabench.cli import _read_reports, make_summary, make_report

    run = runs_folder / "rijubigo.2023-03-24_13:45:27.512446"

    runs = [run]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())

    make_report(summary, None)
