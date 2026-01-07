import json
import pandas as pd
import altair as alt


def main():
    #
    # Estimate the marginal COST
    #
    path = "/home/ciara/benchdevenv/projects/output.csv"

    df = pd.read_csv(path)

    print(df["metric"].unique())
    print(df["run_id"].unique())
    print(df.columns)

    plot_df = (
        df
        .loc[df["run_id"] == "561f143b"]
        .loc[df["metric"].isin(["gpu.power", "rate"])]
        .copy()
    )

    plot_df.to_csv("partial.csv")

    base = alt.Chart(plot_df).encode(
        x=alt.X("time_norm:Q", title="Time (s)")
    )

    power_line = base.transform_filter(
        alt.datum.metric == "gpu.power"
    ).mark_line(color="red").encode(
        y=alt.Y("value:Q", title="GPU Power (W)")
    )

    rate_line = base.transform_filter(
        alt.datum.metric == "rate"
    ).mark_line(color="blue").encode(
        y=alt.Y("value:Q", title="Throughput (rate)", axis=alt.Axis(orient="right"))
    )

    chart = alt.layer(
        power_line,
        rate_line
    ).resolve_scale(
        y="independent"
    )

    chart.save("chart.png")


if __name__ == "__main__":
    main()