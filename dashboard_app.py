"""Entry point của dashboard, gồm hai trang.

    streamlit run dashboard_app.py

File này nằm ở **project root chứ không phải trong `src/dashboard/`**, và đó là chủ ý:
`streamlit run` đặt thư mục chứa script lên `sys.path` chứ không phải thư mục hiện hành.
Nếu để trong `src/dashboard/` thì chỉ có `src/dashboard/` nằm trên path, và mọi
`from src.… import` trong project sẽ hỏng với `ModuleNotFoundError: No module named 'src'`.
Đặt ở root thì project root lên path, các import dùng chung với pipeline chạy bình thường.

Hai trang:

- **Forecast** (`src/dashboard/page_forecast.py`): bản đồ demand theo zone và giờ, gói gọn
  trong một khung hình.
- **Experiment** (`src/dashboard/page_experiment.py`): so sánh kết quả thí nghiệm theo đúng
  cấu trúc README, đọc lại `results/stats/`.

Dùng `st.navigation` với callable thay vì thư mục `pages/`: các trang là module trong
`src/dashboard/` nên không phải lo `sys.path` của từng file trang.
"""

import streamlit as st

from src.dashboard import page_experiment, page_forecast


def main() -> None:
    """Cấu hình trang và điều hướng giữa hai trang."""
    st.set_page_config(
        page_title="Ride demand forecast",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Phải đặt url_path tường minh: hai trang cùng tên hàm `render` nên Streamlit suy ra
    # cùng một pathname và báo lỗi trùng URL.
    navigation = st.navigation(
        [
            st.Page(
                page_forecast.render, title="Forecast", icon="🗺️",
                url_path="forecast", default=True,
            ),
            st.Page(
                page_experiment.render, title="Experiment", icon="📊",
                url_path="experiment",
            ),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
