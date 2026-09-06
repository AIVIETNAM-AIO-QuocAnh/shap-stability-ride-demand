"""Trang Forecast: bản đồ demand theo zone và giờ.

Đọc lại prediction artifact trong `results/`, không train lại gì. Chuỗi hiển thị bằng
tiếng Anh; comment và docstring giữ tiếng Việt cho đồng bộ với các module khác.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.data_access import (
    MODELS,
    VARIANTS,
    attach_zone_names,
    filter_predictions,
    fold_for_date,
    holiday_name,
    load_predictions,
    load_variant_features,
    load_weekly_features,
    load_zone_catalog,
    load_zone_geojson,
    selectable_date_range,
)
from src.dashboard.holidays_theme import theme_for


LAYERS = {
    "Forecast": ("y_pred", "Forecast demand"),
    "Actual": ("y_true", "Actual demand"),
    "Error": ("error", "Error (forecast minus actual)"),
}
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Giải thích ý nghĩa từng variant; danh sách feature lấy động từ config nên phần
# trong ngoặc luôn khớp dữ liệu thật, không sợ chú thích nói một đằng model chạy một nẻo.
# Dòng mô tả phải đủ ngắn để nằm gọn MỘT dòng trong pane filter hẹp: dài hơn là bị ngắt
# dòng giữa chừng, trông rất xấu. Câu đầy đủ nằm ở tooltip của chính slicer Variant.
VARIANT_NOTES = {
    "A": ("Three lags kept separate",
          "A: baseline, each weekly lag stays its own feature."),
    "B": ("Median of the three lags",
          "B: aggregated, the three weekly lags become their median."),
    "C": ("Nearest lag plus median",
          "C: hybrid, keeps the nearest weekly lag and adds the median."),
}

# Màu nhạt cho đường lag vẽ chồng lên biểu đồ 24 giờ: đủ thấy nhưng không tranh
# spotlight với hai đường chính forecast và actual.
LAG_COLORS = ("#9aa7b8", "#b8a99a", "#a99ab8", "#9ab8a7")

# Màu cho cột error: dương nghĩa là dự báo cao hơn thực tế.
ERROR_POSITIVE_COLOR = "#c92a2a"
ERROR_NEGATIVE_COLOR = "#2b8a3e"

# Chiều cao cố định để ba cột cân nhau và dashboard nằm gọn một khung hình.
# Bố cục: cột trái hẹp chứa slicer dữ liệu, cột giữa bản đồ, cột phải bảng và biểu đồ 24 giờ.
#   cột slicer  = 6 control xếp dọc, khoảng 384px sau khi nén nhãn bằng CSS
#   cột bản đồ  = 50 (control Map layer) + MAP_HEIGHT
#   cột phải    = 54 (control Top N) + TABLE_HEIGHT + PROFILE_HEIGHT
#   toàn trang  = 64 padding trên + 52 tiêu đề + 14 divider + 40 hàng badge + cột cao nhất + 10
# Tổng khoảng 647px: vừa cả laptop 1366x768 (vùng hiển thị khoảng 648px).
MAP_HEIGHT = 415
TABLE_HEIGHT = 165
PROFILE_HEIGHT = 248

COMPACT_CSS = """
<style>
  /* padding-top phải đủ lớn để hở thanh header cố định của Streamlit (cao 3.75rem).
     Hạ xuống nhỏ hơn thì hàng đầu tiên chui xuống dưới header và bị che. */
  .block-container {padding: 4rem 1.4rem 0.4rem 1.4rem; max-width: 100%;}
  [data-testid="stElementToolbar"] {display: none;}
  div[data-testid="stVerticalBlock"] {gap: 0.4rem;}
  /* Nén nhãn widget: cột slicer xếp dọc 8 control, nhãn cỡ mặc định làm cột quá cao. */
  [data-testid="stWidgetLabel"] p {font-size: 0.78rem; margin-bottom: 1px;}
  [data-testid="stCaptionContainer"] p {font-size: 0.72rem; margin: 0;}
  h1 {font-size: 1.3rem; margin: 0; padding: 0;}
  h3 {font-size: 0.95rem; margin: 0; padding: 0;}
  hr {margin: 0.3rem 0;}
</style>
"""


def _status_row_html(
    holiday: str | None, weekday: str, date_label: str,
    variant: str, model: str, fold: str, n_zones: int,
) -> str:
    """Một hàng duy nhất: badge ngày bên trái, các chip trạng thái ngay cạnh.

    Toàn bộ bọc trong một `div` **block-level** dùng `display:flex`. Nếu để pill ở dạng
    `inline-flex` trần thì nó là hộp inline: chiều cao hộp lớn hơn line box của thẻ cha
    nên tràn ra ngoài và đè lên phần bên dưới (đã gặp: badge đè lên bản đồ).
    """
    chip = (
        "display:inline-flex;align-items:center;padding:4px 11px;border-radius:999px;"
        "font-size:0.73rem;background:rgba(128,128,128,0.14);white-space:nowrap"
    )
    if holiday:
        theme = theme_for(holiday)
        badge_style = (
            f"background:linear-gradient(110deg,{theme.gradient_from},{theme.gradient_to});"
            f"color:{theme.text_color};box-shadow:0 2px 8px rgba(0,0,0,0.18)"
        )
        emoji, headline = theme.emoji, holiday
        tail = f" · {weekday}, {date_label} · {theme.tagline}"
        decoration = (
            f"<span style='margin-left:10px;font-size:0.85rem;opacity:0.75;"
            f"letter-spacing:2px'>{theme.decoration}</span>"
        )
    else:
        badge_style = (
            "background:rgba(128,128,128,0.12);color:inherit;"
            "border:1px solid rgba(128,128,128,0.25)"
        )
        emoji, headline, tail, decoration = "🗓️", weekday, f", {date_label}", ""

    badge = (
        f"<span style='display:inline-flex;align-items:center;gap:8px;padding:5px 14px;"
        f"border-radius:999px;line-height:1.5;{badge_style}'>"
        f"<span style='font-size:1.1rem'>{emoji}</span>"
        f"<span style='font-size:0.84rem;white-space:nowrap'><b>{headline}</b>"
        f"<span style='opacity:0.85'>{tail}</span></span>{decoration}</span>"
    )
    return (
        "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:8px;"
        "margin:0 0 10px 0;line-height:1.5'>"
        f"{badge}"
        f"<span style='{chip}'>fold&nbsp;<b>{fold}</b></span>"
        f"<span style='{chip}'><b>{n_zones}</b>/50 zones</span>"
        f"<span style='{chip}'>{variant} / {model}</span>"
        "</div>"
    )


def _variant_note_html(variant: str) -> str:
    """Chú thích variant: một dòng mô tả, một dòng danh sách feature.

    Dựng bằng HTML thay vì `st.caption` để kiểm soát `line-height` và khoảng cách với
    slicer ngay bên dưới; `st.caption` để mặc định làm chữ dính sát nhãn "Model".

    Danh sách feature lấy động từ config nên chú thích không thể lệch với dữ liệu thật.
    """
    features = " · ".join(load_variant_features(variant))
    return (
        "<div style='font-size:0.72rem;line-height:1.5;opacity:0.8;margin:-2px 0 12px 2px'>"
        f"{VARIANT_NOTES[variant][0]}<br>"
        "<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        f"font-size:0.68rem;opacity:0.85'>{features}</span>"
        "</div>"
    )


def _filters(panel) -> dict:
    """Toàn bộ slicer xếp dọc trong cột bên trái.

    Gom hết vào một cột thay vì trải ngang phía trên: cột dọc lấy chỗ theo chiều ngang
    (vốn còn thừa trên màn rộng) thay vì chiều dọc (vốn là thứ khan hiếm), nhờ đó cả
    dashboard nằm gọn một khung hình mà không phải cuộn.

    `Map layer` và `Top N` **không** nằm ở đây mà đặt ngay trên bản đồ và bảng: chúng chỉ
    đổi cách hiển thị một thành phần cụ thể, để cạnh thành phần đó thì dễ nối hơn.
    """
    first_date, last_date = selectable_date_range()
    catalog = load_zone_catalog()
    districts = sorted(catalog["borough"].unique())

    selected_date = panel.date_input(
        "Date", value=first_date, min_value=first_date, max_value=last_date, format="DD/MM/YYYY"
    )
    selected_hour = panel.slider("Hour", 0, 23, 8, format="%d:00")
    selected_districts = panel.multiselect("District", options=districts, placeholder="All")

    # Zone chỉ liệt kê trong district đang chọn, để hai bộ lọc không mâu thuẫn nhau.
    in_scope = catalog[catalog["borough"].isin(selected_districts)] if selected_districts else catalog
    zone_labels = {
        int(row.pu_location_id): f"{row.zone_name} ({row.borough})"
        for row in in_scope.itertuples()
    }
    selected_zone_ids = panel.multiselect(
        "Zone",
        options=list(zone_labels),
        format_func=lambda zone_id: zone_labels.get(zone_id, f"Zone {zone_id}"),
        placeholder=f"All {len(zone_labels)} zones",
    )

    # Pane filter hẹp nên xếp dọc, không đặt Variant cạnh Model.
    variant = panel.segmented_control(
        "Variant", VARIANTS, default=VARIANTS[0], selection_mode="single",
        help="  \n".join(long for _, long in VARIANT_NOTES.values()),
    ) or VARIANTS[0]
    # Chú thích ngay dưới slicer Variant để nối được với cái vừa bấm.
    panel.markdown(_variant_note_html(variant), unsafe_allow_html=True)
    model = panel.segmented_control(
        "Model", MODELS, default=MODELS[0], selection_mode="single"
    ) or MODELS[0]

    # Không chọn zone cụ thể thì lấy toàn bộ zone của district đang chọn.
    effective_zone_ids = selected_zone_ids or (list(zone_labels) if selected_districts else [])
    return {
        "date": selected_date,
        "hour": selected_hour,
        "districts": selected_districts,
        "zone_ids": effective_zone_ids,
        "variant": variant,
        "model": model,
    }


def _render_map(hour_frame: pd.DataFrame, layer: str) -> None:
    """Choropleth 50 zone, màu theo lớp đang chọn."""
    column_name, layer_title = LAYERS[layer]
    if hour_frame.empty:
        st.info("No data for the current filters.")
        return

    if column_name == "error":
        bound = float(hour_frame["error"].abs().max()) or 1.0
        color_arguments = {"color_continuous_scale": "RdBu_r", "range_color": (-bound, bound)}
    else:
        # Dùng chung thang màu cho forecast và actual để đổi lớp vẫn so sánh được bằng mắt.
        shared_max = float(max(hour_frame["y_pred"].max(), hour_frame["y_true"].max()))
        color_arguments = {"color_continuous_scale": "YlOrRd", "range_color": (0.0, shared_max)}

    figure = px.choropleth_map(
        hour_frame,
        geojson=load_zone_geojson(),
        locations="pu_location_id",
        featureidkey="properties.pu_location_id",
        color=column_name,
        hover_name="zone_name",
        custom_data=["borough", "y_pred", "y_true", "error"],
        map_style="carto-positron",
        zoom=9.2,
        center={"lat": 40.735, "lon": -73.94},
        opacity=0.78,
        labels={column_name: layer_title},
        **color_arguments,
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b> (%{customdata[0]})<br>"
            "Forecast: %{customdata[1]:.1f}<br>"
            "Actual: %{customdata[2]:.0f}<br>"
            "Error: %{customdata[3]:+.1f}<extra></extra>"
        )
    )
    figure.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=MAP_HEIGHT,
        coloraxis_colorbar={"title": "", "thickness": 12, "len": 0.85},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _color_error(value: float) -> str:
    """Error dương tô đỏ (dự báo cao hơn thực tế), âm tô xanh."""
    color = ERROR_POSITIVE_COLOR if value > 0 else ERROR_NEGATIVE_COLOR
    return f"color:{color};font-weight:600"


def _render_top_zones(hour_frame: pd.DataFrame, top_n: int) -> None:
    """Bảng top N zone theo forecast demand cao nhất.

    Hai cột forecast và actual vẽ thành bar ngay trong cell, **dùng chung một thang**
    (max của cả hai cột) để so được cả giữa các zone lẫn giữa forecast với actual.
    """
    ranked = hour_frame.sort_values("y_pred", ascending=False).head(top_n).copy()
    ranked.insert(0, "#", range(1, len(ranked) + 1))
    display = ranked[["#", "zone_name", "y_pred", "y_true", "error"]].rename(
        columns={"zone_name": "zone", "y_pred": "forecast", "y_true": "actual", "error": "error"}
    )
    bar_max = float(max(display["forecast"].max(), display["actual"].max())) or 1.0
    styled = display.style.map(_color_error, subset=["error"])
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        height=TABLE_HEIGHT,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "forecast": st.column_config.ProgressColumn(
                format="%.1f", min_value=0.0, max_value=bar_max
            ),
            "actual": st.column_config.ProgressColumn(
                format="%d", min_value=0.0, max_value=bar_max
            ),
            "error": st.column_config.NumberColumn(format="%+.1f"),
        },
    )


def _render_daily_profile(
    day_frame: pd.DataFrame, feature_frame: pd.DataFrame, variant: str, selected_hour: int
) -> None:
    """Đường 24 giờ của **toàn bộ zone đang chọn**, forecast chồng actual.

    Cộng dồn theo giờ thay vì khoá vào một zone: biểu đồ luôn khớp đúng phạm vi mà bộ lọc
    District/Zone ở trên đang mô tả. Muốn xem riêng một zone thì chọn zone đó ở bộ lọc.

    Vẽ thêm các weekly feature của chính variant đang chọn (lag_168, median_lag_3w, ...)
    bằng nét đứt màu nhạt: chúng là thứ model nhìn vào để dự báo, nên đặt cạnh nhau thì
    thấy được model đang bám theo tuần nào, mà vẫn không tranh spotlight với forecast
    và actual.
    """
    if day_frame.empty:
        return
    n_zones = int(day_frame["pu_location_id"].nunique())
    if n_zones == 1:
        title = f"{day_frame['zone_name'].iloc[0]} over 24 hours"
    else:
        title = f"All {n_zones} selected zones, total demand over 24 hours"

    hourly = day_frame.groupby("hour", as_index=False)[["y_pred", "y_true"]].sum()
    hourly = hourly.rename(columns={"y_pred": "Forecast", "y_true": "Actual"})

    lag_names = list(load_variant_features(variant))
    if not feature_frame.empty:
        lag_hourly = feature_frame.groupby("hour", as_index=False)[lag_names].sum()
        hourly = hourly.merge(lag_hourly, on="hour", how="left")

    value_columns = ["Forecast", "Actual"] + [name for name in lag_names if name in hourly.columns]
    long_frame = hourly.melt(
        id_vars="hour", value_vars=value_columns, var_name="series", value_name="demand"
    )
    color_map = {"Forecast": "#d95f02", "Actual": "#1b9e77"}
    for index, name in enumerate(lag_names):
        color_map[name] = LAG_COLORS[index % len(LAG_COLORS)]

    figure = px.line(
        long_frame, x="hour", y="demand", color="series",
        color_discrete_map=color_map,
        labels={"hour": "", "demand": "", "series": ""},
        title=title,
    )
    for trace in figure.data:
        if trace.name in lag_names:
            # Nét đứt, mảnh, mờ: đủ đọc xu hướng mà không kéo mắt khỏi hai đường chính.
            trace.update(mode="lines", opacity=0.55, line={"dash": "dot", "width": 1.6})
        else:
            trace.update(mode="lines+markers", line={"width": 2.4}, marker={"size": 5})

    figure.add_vline(x=selected_hour, line_dash="dot", line_color="#888")
    # Legend đặt DƯỚI vùng vẽ, không đặt trên: tiêu đề và legend cùng nằm trong dải margin
    # phía trên nên đâm vào nhau khi tiêu đề dài. Bỏ luôn nhãn trục x ("hour") vì các mốc
    # 0..24 đã tự rõ, lấy chỗ đó cho legend. Margin dưới phải đủ chứa cả nhãn mốc lẫn legend,
    # vì Plotly không tự nới margin đã khai báo tường minh.
    figure.update_layout(
        margin={"r": 4, "t": 28, "l": 4, "b": 60},
        height=PROFILE_HEIGHT,
        title={"font": {"size": 14}, "x": 0, "y": 0.97},
        xaxis={"dtick": 2, "title": None},
        legend={
            "orientation": "h", "yanchor": "top", "y": -0.16,
            "xanchor": "center", "x": 0.5, "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render() -> None:
    """Dựng trang Forecast, gói trong một khung hình."""
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)

    # Giữ chỗ cho tiêu đề và hàng badge trước khi vẽ slicer, để chúng nằm trên cùng dù
    # nội dung (fold, số zone, ngày lễ) chỉ biết được sau khi đọc lựa chọn người dùng.
    header_slot = st.container()
    status_slot = st.container()

    filter_column, map_column, side_column = st.columns([1.15, 2.6, 2.25], gap="medium")

    choice = _filters(filter_column)
    fold = fold_for_date(choice["date"])
    predictions = load_predictions(choice["variant"], fold, choice["model"])
    day_frame = attach_zone_names(
        filter_predictions(predictions, selected_date=choice["date"], zone_ids=choice["zone_ids"])
    )
    hour_frame = day_frame[day_frame["hour"] == choice["hour"]]

    with header_slot:
        st.title("Ride demand forecast by zone and hour")
        st.divider()

    with status_slot:
        st.markdown(
            _status_row_html(
                holiday_name(choice["date"]),
                WEEKDAY_NAMES[choice["date"].weekday()],
                choice["date"].strftime("%d %b %Y"),
                choice["variant"], choice["model"], fold,
                int(hour_frame["pu_location_id"].nunique()),
            ),
            unsafe_allow_html=True,
        )

    with map_column:
        layer = st.segmented_control(
            "Map layer", tuple(LAYERS), default="Forecast", selection_mode="single"
        ) or "Forecast"
        _render_map(hour_frame, layer)

    with side_column:
        n_available = int(hour_frame["pu_location_id"].nunique()) or 1
        top_n = int(st.number_input(
            "Top N zones", min_value=1, max_value=n_available,
            value=min(10, n_available), step=1,
        ))
        _render_top_zones(hour_frame, top_n)
        feature_frame = load_weekly_features(fold, choice["variant"])
        feature_frame = feature_frame[feature_frame["date"] == choice["date"]]
        if choice["zone_ids"]:
            feature_frame = feature_frame[
                feature_frame["pu_location_id"].isin(choice["zone_ids"])
            ]
        _render_daily_profile(day_frame, feature_frame, choice["variant"], choice["hour"])
