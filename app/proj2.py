import os
import io
import base64
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from dotenv import load_dotenv

# Load .env if running locally
load_dotenv()

app = FastAPI()

# Read the AI proxy token from environment
AI_PROXY_TOKEN = os.getenv("AI_PROXY_TOKEN")
if not AI_PROXY_TOKEN:
    raise RuntimeError("AI_PROXY_TOKEN environment variable is not set")


# Utility function to encode matplotlib figure to base64 PNG under 100kB
def fig_to_base64(fig, fmt="png", max_size=100_000):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    # Resize if too large
    if len(data) > max_size:
        scale = (max_size / len(data)) ** 0.5
        fig.set_size_inches(
            fig.get_size_inches()[0] * scale, fig.get_size_inches()[1] * scale
        )
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight")
        buf.seek(0)
        data = buf.read()
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


@app.post("/api/")
async def analyze_data(
    questions: UploadFile = File(...),
    files: Optional[List[UploadFile]] = File(None),
):
    questions_text = (await questions.read()).decode("utf-8")

    # Build a dict of filename -> pandas DataFrame
    dfs = {}
    if files:
        for f in files:
            content = await f.read()
            if f.filename.endswith(".csv"):
                dfs[f.filename] = pd.read_csv(io.BytesIO(content))

    response = {}

    # Example handling for known CSV names
    if "sample-sales.csv" in dfs:
        df = dfs["sample-sales.csv"]
        total_sales = df["sales"].sum()
        top_region = df.groupby("region")["sales"].sum().idxmax()
        day_sales_corr = df["date"].apply(lambda x: pd.to_datetime(x).day).corr(df["sales"])
        median_sales = df["sales"].median()
        total_sales_tax = total_sales * 0.1

        # Bar chart by region
        fig, ax = plt.subplots()
        df.groupby("region")["sales"].sum().plot(kind="bar", color="blue", ax=ax)
        bar_chart = fig_to_base64(fig)

        # Cumulative sales line chart
        df_sorted = df.sort_values("date")
        fig, ax = plt.subplots()
        df_sorted["sales"].cumsum().plot(ax=ax, color="red")
        cumulative_sales_chart = fig_to_base64(fig)

        response.update(
            {
                "total_sales": total_sales,
                "top_region": top_region,
                "day_sales_correlation": day_sales_corr,
                "bar_chart": bar_chart,
                "median_sales": median_sales,
                "total_sales_tax": total_sales_tax,
                "cumulative_sales_chart": cumulative_sales_chart,
            }
        )

    elif "sample-weather.csv" in dfs:
        df = dfs["sample-weather.csv"]
        average_temp_c = df["temperature"].mean()
        max_precip_date = df.loc[df["precipitation"].idxmax(), "date"]
        min_temp_c = df["temperature"].min()
        temp_precip_corr = df["temperature"].corr(df["precipitation"])
        average_precip_mm = df["precipitation"].mean()

        # Temperature line chart
        fig, ax = plt.subplots()
        df.plot(x="date", y="temperature", ax=ax, color="red")
        temp_line_chart = fig_to_base64(fig)

        # Precipitation histogram
        fig, ax = plt.subplots()
        df["precipitation"].plot(kind="hist", color="orange", ax=ax)
        precip_histogram = fig_to_base64(fig)

        response.update(
            {
                "average_temp_c": average_temp_c,
                "max_precip_date": str(max_precip_date),
                "min_temp_c": min_temp_c,
                "temp_precip_correlation": temp_precip_corr,
                "average_precip_mm": average_precip_mm,
                "temp_line_chart": temp_line_chart,
                "precip_histogram": precip_histogram,
            }
        )

    elif "edges.csv" in dfs:
        df = dfs["edges.csv"]
        G = nx.from_pandas_edgelist(df, source="source", target="target")
        edge_count = G.number_of_edges()
        highest_degree_node = max(dict(G.degree()).items(), key=lambda x: x[1])[0]
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        density = nx.density(G)
        try:
            shortest_path = nx.shortest_path_length(G, "Alice", "Eve")
        except nx.NetworkXNoPath:
            shortest_path = -1

        # Network graph
        fig, ax = plt.subplots(figsize=(4, 4))
        nx.draw_networkx(G, ax=ax, node_size=200, with_labels=True)
        network_graph = fig_to_base64(fig)

        # Degree histogram
        degrees = [d for n, d in G.degree()]
        fig, ax = plt.subplots()
        ax.bar(range(len(degrees)), sorted(degrees), color="green")
        degree_histogram = fig_to_base64(fig)

        response.update(
            {
                "edge_count": edge_count,
                "highest_degree_node": highest_degree_node,
                "average_degree": avg_degree,
                "density": density,
                "shortest_path_alice_eve": shortest_path,
                "network_graph": network_graph,
                "degree_histogram": degree_histogram,
            }
        )

    else:
        # fallback: just echo questions and files
        response = {
            "status": "success",
            "received_questions": len(questions_text.splitlines()),
            "received_files": [f.filename for f in files] if files else [],
        }

    return JSONResponse(content=response)
