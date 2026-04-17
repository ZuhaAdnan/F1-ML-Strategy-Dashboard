
import fastf1
from pathlib import Path
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
Path("f1_cache").mkdir(exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")
#race data
YEAR = 2023
RACE = "Monza"
session = fastf1.get_session(YEAR, RACE, "R")
session.load()
laps = session.laps.pick_quicklaps().reset_index()
laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
#driver name abbreviations
driver_map = {}
for d in laps["Driver"].unique():
    try:
        driver_map[d] = session.get_driver(d)["Abbreviation"]
    except:
        driver_map[d] = d
laps["Driver"] = laps["Driver"].map(driver_map)
laps["Compound"] = laps["Compound"].str.upper()
max_lap = int(laps["LapNumber"].max())
TIRE_COLORS = {"SOFT": "#ff4d4d", "MEDIUM": "#ffdb4d", "HARD": "#4da6ff"}
MAX_STINT = {"SOFT": 15, "MEDIUM": 25, "HARD": 40}
ml_data = laps.dropna(subset=["LapTimeSec"]).copy()
le_driver = LabelEncoder()
le_compound = LabelEncoder()
ml_data["Driver_enc"] = le_driver.fit_transform(ml_data["Driver"])
ml_data["Compound_enc"] = le_compound.fit_transform(ml_data["Compound"])
ml_data["StintStart"] = ml_data.groupby(
    ["Driver", "Compound"]
)["LapNumber"].transform("min")
ml_data["TireAge"] = ml_data["LapNumber"] - ml_data["StintStart"] + 1
#labels
ml_data["PositionChange"] = ml_data.groupby("Driver")["Position"].diff().fillna(0)
ml_data["Overtake"] = (ml_data["PositionChange"] < 0).astype(int)
ml_data["NextPosition"] = ml_data.groupby("Driver")["Position"].shift(-1)
def pit_label(row):
    max_stint = MAX_STINT.get(row["Compound"], 20)
    if row["TireAge"] >= 0.9 * max_stint:
        return "PIT_NOW"
    elif row["TireAge"] >= 0.7 * max_stint:
        return "PIT_SOON"
    return "STAY_OUT"
ml_data["PitLabel"] = ml_data.apply(pit_label, axis=1)
#ml model training
FEATURES = ["Driver_enc", "Compound_enc", "TireAge"]
X = ml_data[FEATURES]
#lap time prediction
lap_model = RandomForestRegressor(n_estimators=200, random_state=42)
lap_model.fit(X, ml_data["LapTimeSec"])
#pit decision
pit_model = RandomForestClassifier(n_estimators=200, random_state=42)
pit_model.fit(X, ml_data["PitLabel"])
#overtake probability
overtake_model = RandomForestClassifier(n_estimators=150, random_state=42)
overtake_model.fit(X, ml_data["Overtake"])
#postpit position
pos_data = ml_data.dropna(subset=["NextPosition"])
pos_model = RandomForestRegressor(n_estimators=200, random_state=42)
pos_model.fit(pos_data[FEATURES], pos_data["NextPosition"])
#dashboard
app = Dash(__name__)
app.layout = html.Div(style={"backgroundColor": "#111", "color": "white"}, children=[
    html.H1("F1 ML Strategy Wall", style={"textAlign": "center"}),
    dcc.Slider(
        id="lap-slider",
        min=1,
        max=max_lap,
        value=1,
        step=1,
        marks={i: str(i) for i in range(1, max_lap + 1, 5)}
    ),
    dcc.Dropdown(
        id="driver-dropdown",
        options=[{"label": "All Drivers", "value": "ALL"}] +
                [{"label": d, "value": d} for d in sorted(laps["Driver"].unique())],
        value="ALL",
        clearable=False,
        style={"width": "300px", "marginTop": "10px"}
    ),
    dcc.Interval(id="interval", interval=8000, n_intervals=0),
    html.Div(style={"display": "flex", "marginTop": "20px"}, children=[
        html.Div(style={"width": "50%"}, children=[
            dcc.Graph(id="position-chart"),
            dcc.Graph(id="lap-chart")
        ]),
        html.Div(style={"width": "50%", "padding": "10px"}, children=[
            dash_table.DataTable(
                id="strategy-table",
                columns=[{"name": c, "id": c} for c in [
                    "Driver", "Pos", "Tyre", "Tyre Age",
                    "Pred Lap", "Pit ML",
                    "Overtake %", "Pred Pos After Pit"
                ]],
                style_cell={"backgroundColor": "#222", "color": "white"},
                style_header={"backgroundColor": "#333"},
                page_size=20
            )
        ])
    ])
])
@app.callback(
    Output("lap-slider", "value"),
    Input("interval", "n_intervals"),
    Input("lap-slider", "value")
)
def auto_lap(_, lap):
    return min(lap + 1, max_lap)
#update dashboard
@app.callback(
    Output("position-chart", "figure"),
    Output("lap-chart", "figure"),
    Output("strategy-table", "data"),
    Input("lap-slider", "value"),
    Input("driver-dropdown", "value")
)
def update(lap, driver_sel):
    current = laps[laps["LapNumber"] <= lap]
    latest = current.sort_values("LapNumber").groupby("Driver").tail(1)
    latest = latest.sort_values("Position")
    #position chart
    disp = latest if driver_sel == "ALL" else latest[latest["Driver"] == driver_sel]
    fig_pos = go.Figure(go.Bar(
        x=[max_lap - p + 1 for p in disp["Position"]],
        y=disp["Driver"],
        orientation="h",
        marker_color=[TIRE_COLORS.get(c, "#888") for c in disp["Compound"]],
        text=disp["Position"]
    ))
    fig_pos.update_layout(
        title=f"Positions – Lap {lap}",
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font_color="white",
        yaxis=dict(autorange="reversed")
    )
    #lap time chart
    fig_lap = go.Figure()
    drivers = [driver_sel] if driver_sel != "ALL" else latest["Driver"]
    for d in drivers:
        d_laps = current[current["Driver"] == d]
        fig_lap.add_trace(go.Scatter(
            x=d_laps["LapNumber"],
            y=d_laps["LapTimeSec"],
            mode="lines",
            name=d
        ))
    fig_lap.update_layout(
        title="Lap Times",
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font_color="white"
    )
    #strategy table
    rows = []
    for _, r in latest.iterrows():
        d, c = r["Driver"], r["Compound"]
        d_laps = current[current["Driver"] == d]
        stint_start = d_laps[d_laps["Compound"] == c]["LapNumber"].min()
        tire_age = lap - stint_start + 1
        X_pred = pd.DataFrame([{
            "Driver_enc": le_driver.transform([d])[0],
            "Compound_enc": le_compound.transform([c])[0],
            "TireAge": tire_age
        }])
        pred_lap = lap_model.predict(X_pred)[0]
        pit_dec = pit_model.predict(X_pred)[0]
        overtake_prob = overtake_model.predict_proba(X_pred)[0][1]
        pred_pos = pos_model.predict(X_pred)[0]
        rows.append({
            "Driver": d,
            "Pos": r["Position"],
            "Tyre": c,
            "Tyre Age": tire_age,
            "Pred Lap": round(pred_lap, 2),
            "Pit ML": pit_dec,
            "Overtake %": f"{overtake_prob*100:.1f}%",
            "Pred Pos After Pit": int(round(pred_pos))
        })

    if driver_sel != "ALL":
        rows = [x for x in rows if x["Driver"] == driver_sel]

    return fig_pos, fig_lap, rows

if __name__ == "__main__":
    app.run(debug=True)
