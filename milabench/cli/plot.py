


def cli_report_plot():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, help="run folder")
    args = parser.parse_args()

    from .report import generate_flat_report

    df = generate_flat_report(args)




def power_over_time(df):
    # df = df[df["power"] == '700']

    df["clock"] = df["clock"].fillna('1785')
    df["power"] = df["power"].fillna('600')

    df = df[df["clock"] == '1785']
    df = df[df["metric"] == "gpudata.power"]
    
    df.to_csv("power_only.csv")

    import altair as alt

    chart = alt.Chart(df).mark_point().encode(
        x=alt.X("time_norm:Q", title="Time since start (s)"),
        y=alt.Y("value:Q", title="Metric Value"),
        color=alt.Color("power:O", title="Power"), 
    ).properties(
        width=500,
        height=500
    ).facet(
        column=alt.Column("bench:N", title="Bench"),
        row=alt.Row("observation:N", title="observation")
    ).resolve_scale(
        x='independent',
        y='independent'
    )

    chart.save("power_evol.png")


def arguments():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--selection")

    return parser.parse_args()



class Selector:
    def __init__(self, args):
        self.args = args

    def __call__(self, filepath, meta):
        # if meta["bench"].startswith("vllm-sweep"):
        #     return True

        # if meta["bench"] in ("txt-to-image-gpus",):
        #     return True


        if meta["bench"] in ("convnext_large-fp16",):
            return True

        return False


def cli_event_plot():
    import pandas as pd

    from ..report.read import extract_milabench_metrics, augment_energy_estimator

    args= arguments()

    df = pd.DataFrame(
        augment_energy_estimator(extract_milabench_metrics(args.folder, Selector(args)))
    )


    print(df)
    print(df["clock"].unique())
    print(df["metric"].unique())
    print(df.columns)

    df["time_norm"] = (
        df["time"] - df[df["metric"] == "gpudata.power"].groupby("run_id")["time"].transform("min")
    )


    power_over_time(df)



if __name__ == "__main__":
    cli_event_plot()
