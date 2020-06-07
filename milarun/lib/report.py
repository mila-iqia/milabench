from hrepr import HTML
from collections import defaultdict
from functools import reduce
import json
import os
import numpy as np
import glob
import operator
import pandas as pd
from pandas import DataFrame, Series
import math
from itertools import chain


H = HTML()


def extract_reports(report_folder, report_names):
    filenames = chain(
        glob.glob(f'{report_folder}/**/{report_names}.*.json'),
        glob.glob(f'{report_folder}/{report_names}.*.json')
    )
    reports = defaultdict(list)
    for filename in filenames:
        entry = json.load(open(filename, 'r'))
        entry['__path__'] = filename
        reports[entry['name']].append(entry)
    return reports


def _report_nans(reports, nans=None):
    nans = [] if nans is None else nans
    for name, entries in reports.items():
        for entry in entries:
            # if any(math.isnan(x) for x in entry['batch_loss']):
            if "loss" in entry and math.isnan(entry["loss"]):
                entry['failure'] = 'A nan was found in batch_loss'
                entries.remove(entry)
                nans.append(entry)
    return nans


def _report_failures(reports, fails=None):
    fails = [] if fails is None else fails
    for name, entries in reports.items():
        for entry in list(entries):
            if 'success' in entry and not entry['success']:
                entry['failure'] = 'The test did not succeed'
                entries.remove(entry)
                fails.append(entry)
    return fails


def _report_pergpu(baselines, reports, measure='mean'):
    results = defaultdict(lambda: defaultdict(list))
    all_devices = set()
    for name, entries in reports.items():
        for entry in entries:
            devices = [int(device_id) for device_id in entry['vcd'].split(',')]
            if len(devices) == 1:
                device, = devices
                all_devices.add(device)
                results[name][device].append(entry['train_item']['avg'])

    all_devices = list(sorted(all_devices))

    results = {
        name: {device: getattr(Series(data), measure)()
               for device, data in device_results.items()}
        for name, device_results in results.items()
    }
    results = {
        k: v for k, v in
        sorted(results.items(), key=lambda x: x[0] or "_")
    }

    df = DataFrame(results).transpose()
    df = df.reindex(columns=all_devices)

    maxes = df.loc[:, all_devices].max(axis=1).transpose()
    df = (df.transpose() / maxes).transpose()

    return df


_table_style = H.style("""
body {
    font-family: monospace;
}
td, th {
    text-align: right;
    min-width: 75px;
}
.result-PASS {
    color: green;
    font-weight: bold;
}
.result-FAIL {
    color: red;
    font-weight: bold;
}
.score, .rpp {
    color: blue;
    font-weight: bold;
}
""")


def _style(df):

    def _redgreen(value):
        return 'color: green' if value else 'color: red'

    def _gpu_pct(value):
        if value >= 0.9:
            color = '#080'
        elif value >= 0.8:
            color = '#880'
        elif value >= 0.7:
            color = '#F80'
        else:
            color = '#F00'
        return f'color: {color}'

    def _perf(values):
        return (values >= df['perf_tgt']).map(_redgreen)

    def _sd(values):
        return (values <= df['sd%_tgt']).map(_redgreen)

    # Text formatting
    sty = df.style
    sty = sty.format(_formatters)

    # Format GPU efficiency map columns
    gpu_columns = set(range(16)) & set(df.columns)
    sty = sty.applymap(_gpu_pct, subset=list(gpu_columns))

    # Format perf column
    if 'perf' in df.columns and 'perf_tgt' in df.columns:
        sty = sty.apply(_perf, subset=['perf'])
        sty = sty.applymap(
            lambda x: 'font-weight: bold' if x >= 1 else '',
            subset=['perf']
        )

    # Format sd% column
    if 'sd%' in df.columns and 'sd%_tgt' in df.columns:
        sty = sty.apply(_sd, subset=['sd%'])

    # Format pass/fail column
    for col in ['pass', 'sd%_pass', 'perf_pass']:
        if col in df.columns:
            sty = sty.applymap(_redgreen, subset=[col])

    return sty


def _display_title(file, title, stdout_display=True):
    if stdout_display:
        print()
        print('=' * len(title))
        print(title)
        print('=' * len(title))
        print()

    if file:
        print(H.h2(title), file=file)


def _display_table(file, title, table, stdout_display=True):
    _display_title(file, title, stdout_display=stdout_display)

    if stdout_display:
        print(table.to_string(formatters=_formatters))

    if file:
        sty = _style(table)
        print(sty._repr_html_(), file=file)


def _report_global(baselines, reports):
    for name, entries in sorted(
        reports.items(),
        key=lambda x: x[0] or "_"
    ):
        print("---", name, len(entries))
        for x in entries:
            print(len(x["timings"]["train"]["times"]))

    df = DataFrame(
        {
            name: {
                'n': len(entries),
                'mean': Series(
                    reduce(operator.add, [
                        x["timings"]["train"]["times"]
                        for x in entries
                    ])
                ).mean(),
                'sd': Series(
                    reduce(operator.add, [
                        x["timings"]["train"]["times"]
                        for x in entries
                    ])
                ).std(),
            }
            for name, entries in sorted(
                reports.items(),
                key=lambda x: x[0] or "_"
            )
            if isinstance(name, str) and len(entries) > 0
        }
    ).transpose()
    has_baselines = baselines is not None
    if has_baselines:
        baselines = DataFrame(baselines).transpose()
        df['target'] = baselines['target']
        df['perf_tgt'] = baselines['perf_target']
        df['sd%_tgt'] = baselines['sd_target']
        df['perf'] = df['mean'] / df['target']
    df['sd%'] = df['sd'] / df['mean']
    if has_baselines:
        df['perf_pass'] = df['perf'] >= df['perf_tgt']
        df['sd%_pass'] = df['sd%'] <= df['sd%_tgt']
        df['pass'] = df['perf_pass'] & df['sd%_pass']
    return df


def passfail(x):
    return 'PASS' if x else 'FAIL'


_formatters = {
    'n': '{:.0f}'.format,
    'target': '{:10.2f}'.format,
    'sd%_tgt': '{:10.0%}'.format,
    'mean': '{:10.2f}'.format,
    'sd': '{:10.2f}'.format,
    'perf': '{:10.2f}'.format,
    'sd%': '{:10.1%}'.format,
    'pass': passfail,
    'sd%_pass': passfail,
    'perf_pass': passfail,
    0: '{:.0%}'.format,
    1: '{:.0%}'.format,
    2: '{:.0%}'.format,
    3: '{:.0%}'.format,
    4: '{:.0%}'.format,
    5: '{:.0%}'.format,
    6: '{:.0%}'.format,
    7: '{:.0%}'.format,
    8: '{:.0%}'.format,
    9: '{:.0%}'.format,
    10: '{:.0%}'.format,
    11: '{:.0%}'.format,
    12: '{:.0%}'.format,
    13: '{:.0%}'.format,
    14: '{:.0%}'.format,
    15: '{:.0%}'.format,
}


def _format_df(df):
    return df.to_string(formatters=_formatters)


def generate_report(args):
    reports = extract_reports(args.reports, args.jobs)
    nreports = sum(len(entries) for entries in reports.values())

    failures = []

    # Fail check
    _report_failures(reports, failures)
    _report_nans(reports, failures)

    nfailures = len(failures)

    baselines = args.baselines

    outd = True
    title = args.title or args.html.replace('.html', '')

    html = args.html and open(args.html, 'w')
    print(f'<html><head><title>{title}</title></head><body>', file=html)
    print(_table_style, file=html)

    _display_title(
        title=f'{title} ({args.reports})',
        file=html,
        stdout_display=outd
    )

    df_global = _report_global(baselines, reports)
    df = df_global.reindex(
        columns=(
            'n',
            'target',
            'perf_tgt',
            'sd%_tgt',
            'mean',
            # 'sd',
            'perf',
            'perf_pass',
            'sd%',
            'sd%_pass',
            # 'pass',
        )
    )
    _display_table(
        title=f'Results',
        table=df,
        file=html,
        stdout_display=outd,
    )

    fail_ratio = nfailures / nreports
    fail_criterion = fail_ratio <= 0.01

    if failures:
        fail_crit_pf = passfail(fail_criterion)

        _display_title(title='Failures', file=html, stdout_display=outd)

        ratio = nfailures / nreports
        msg = f'{nfailures} failures / {nreports} results ({fail_ratio:.2%})'
        print(msg, '--', fail_crit_pf)
        print()
        print(H.div(msg, " ", H.span[f'result-{fail_crit_pf}'](fail_crit_pf)),
              file=html)
        print(H.br(), file=html)

        for entry in failures:
            message = f'Failure in {entry["__path__"]}: {entry["failure"]}'
            print(message)
            print(H.div(message), file=html)

    # _display_title(title='Global performance', file=html, stdout_display=outd)

    # try:
    #     df_perf = df.drop('scaling')
    # except KeyError:
    #     df_perf = df
    # perf = (df_perf['perf'].prod()) ** (1/df_perf['perf'].count())
    # minreq = 0.9
    # success = perf > minreq

    # pm = df_global['perf_pass'].sum() / df_global['perf_pass'].count()
    # st = df_global['sd%_pass'].sum() / df_global['sd%_pass'].count()
    # xpm = 2 * pm
    # xperf = 5 * perf
    # xst = 1 * st
    # grade = fail_criterion * (xpm + xst + xperf)
    # grade_success = True
    # if args.price:
    #     if grade > 0:
    #         rpp = args.price / grade
    #         rpp = f'{rpp:.2f}'
    #     else:
    #         rpp = 'n/a'
    # else:
    #     rpp = 'n/a'

    # if outd:
    #     print(f'Well functioning score (BF):     {fail_criterion:.2f}')
    #     print(f'Minimal performance score (PM):  {pm:.2f}')
    #     print(f'Standard deviation score (ST):   {st:.2f}')
    #     print(f'Mean performance (geomean) (PG): {perf:.2f}')
    #     print(f'Score (BF(2PM + ST + 5PG)):      {grade:.2f}')
    #     if args.price:
    #         print(f'Price:                           ${args.price:.2f}')
    #         print(f'RPP (Price/Score):               {rpp}')

    # if html:
    #     tb = H.table(
    #         H.tr(
    #             H.th('Well functioning score (BF)'),
    #             H.td(f'{fail_criterion:.2f}'),
    #         ),
    #         H.tr(
    #             H.th('Minimal performance score (PM)'),
    #             H.td(f'{pm:.2f}'),
    #         ),
    #         H.tr(
    #             H.th('Standard deviation score (ST)'),
    #             H.td(f'{st:.2f}'),
    #         ),
    #         H.tr(
    #             H.th('Mean performance (geomean) (PG)'),
    #             H.td(f'{perf:.2f}'),
    #         ),
    #         H.tr(
    #             H.th('Score (BF(2PM + ST + 5PG))'),
    #             H.td[f'XXresult-{passfail(grade_success)}']['score'](
    #                 f'{grade:.2f}'
    #             ),
    #         ),
    #     )
    #     if args.price:
    #         tb = tb(
    #             H.tr(
    #                 H.th('Price'),
    #                 H.td(
    #                     H.span['price'](f'${args.price:.2f}')
    #                 ),
    #             ),
    #             H.tr(
    #                 H.th('RPP (Price/Score)'),
    #                 H.td[f'XXresult-{passfail(grade_success)}'](
    #                     H.span['rpp'](f'{rpp}')
    #                 ),
    #             ),
    #         )
    #     print(tb, file=html)

    # for measure in ['mean', 'min', 'max']:
    #     df = _report_pergpu(baselines, reports, measure=measure)
    #     _display_table(
    #         title=f'Relative GPU performance ({measure})',
    #         table=df,
    #         file=html,
    #         stdout_display=outd,
    #     )

    html.write('</body></html>')
