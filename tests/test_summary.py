from milabench.validation.validation import Summary
from milabench.cli import main


def test_report(runs_folder, capsys, file_regression):
    folder = runs_folder / "rijubigo.2023-03-24_13:45:27.512446"
    try:
        main(["report", "--runs", folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(folder), "XXX")
    file_regression.check(output)


def test_summary(file_regression):
    benchs = ["matmult", "matsub"]
    points = [
        "1. Errors stuff happened",
        "2. Errors stuff happened",
        "3. Errors stuff happened",
    ]
    report = Summary()

    with report.section("Errors"):
        for bench in benchs:
            with report.section(bench):
                for p in points:
                    report.add(p)

    output = ""

    def get_output(data):
        nonlocal output
        output = data

    report.show()
    report.show(get_output)
    file_regression.check(output)


def test_empty_summary():
    points = [
        "1. Errors stuff happened",
        "2. Errors stuff happened",
        "3. Errors stuff happened",
    ]
    report = Summary()

    with report.section("Errors"):
        with report.section("Bench"):
            pass

    output = ""

    def get_output(data):
        nonlocal output
        output = data

    report.show(get_output)

    assert output.strip() == ""


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


def test_summary_per_gpu(runs_folder):
    from milabench.cli import _read_reports, make_summary, make_report

    # run = runs_folder / "MI250.2023-05-08_17_54_51.224604"
    run = runs_folder / "8xA100-SXM-80Go.2023-05-10_13_37_18.387537"
    runs = [run]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())

    make_report(summary, None, mode="per_gpu")


def test_summary_full(runs_folder):
    from milabench.cli import _read_reports, make_summary, make_report

    run = runs_folder / "MI250.2023-05-08_17_54_51.224604"
    run = runs_folder / "8xA100-SXM-80Go.2023-05-10_13_37_18.387537"

    runs = [run]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())

    make_report(summary, None, mode="full")
