"""Bảng màu và biểu tượng riêng cho từng ngày lễ liên bang Mỹ.

Tên khoá lấy đúng chuỗi mà `pandas.tseries.holiday.USFederalHolidayCalendar` trả về,
để `data_access.holiday_name()` tra thẳng được, không cần ánh xạ trung gian.

Mỗi theme mô tả bằng gradient hai màu cộng một màu chữ, chọn theo biểu tượng văn hoá
của chính ngày lễ đó: Giáng sinh xanh thông và đỏ, Thanksgiving vàng hổ phách, Juneteenth
đỏ đen xanh lá, Quốc khánh xanh đỏ trắng, và tương tự.

Chuỗi hiển thị trên dashboard đều bằng tiếng Anh.
"""

from typing import NamedTuple


class HolidayTheme(NamedTuple):
    """Cách trình bày một ngày lễ trên dashboard."""

    emoji: str
    gradient_from: str
    gradient_to: str
    text_color: str
    tagline: str
    decoration: str


DEFAULT_THEME = HolidayTheme(
    "📅", "#495057", "#212529", "#f8f9fa", "US federal holiday", "✦"
)

HOLIDAY_THEMES: dict[str, HolidayTheme] = {
    "New Year's Day": HolidayTheme(
        "🎆", "#12123a", "#3b1d5e", "#ffe08a",
        "Late-night demand spikes around midnight", "🎇 ✨ 🥂",
    ),
    "Birthday of Martin Luther King, Jr.": HolidayTheme(
        "✊", "#2d1b4e", "#6b3fa0", "#ffd873",
        "Federal holiday, reduced commuting", "🕊️ ✊ 🕊️",
    ),
    "Washington's Birthday": HolidayTheme(
        "🎩", "#0d2149", "#7b1e28", "#f2f2f2",
        "Presidents' Day, long weekend travel", "🇺🇸 🎩 🇺🇸",
    ),
    "Memorial Day": HolidayTheme(
        "🎖️", "#14213d", "#8c1c13", "#f8f9fa",
        "Long weekend, city outflow", "🌺 🎖️ 🌺",
    ),
    "Juneteenth National Independence Day": HolidayTheme(
        "✊🏾", "#8c1c13", "#1b4332", "#ffd166",
        "Federal holiday since 2021", "⭐ ✊🏾 ⭐",
    ),
    "Independence Day": HolidayTheme(
        "🎇", "#0a2472", "#9d0208", "#ffffff",
        "Fireworks crowds, sharp evening peak", "🇺🇸 🎆 🗽",
    ),
    "Labor Day": HolidayTheme(
        "🛠️", "#1d3557", "#457b9d", "#f1faee",
        "End of summer, long weekend", "⚙️ 🛠️ ⚙️",
    ),
    "Columbus Day": HolidayTheme(
        "⛵", "#023047", "#0077b6", "#caf0f8",
        "Partial holiday, mixed commuting", "🌊 ⛵ 🧭",
    ),
    "Veterans Day": HolidayTheme(
        "🎖️", "#283618", "#1d3557", "#fefae0",
        "Parades in Manhattan", "🌾 🎖️ 🕊️",
    ),
    "Thanksgiving Day": HolidayTheme(
        "🦃", "#7f4f24", "#dda15e", "#fff3e2",
        "Heaviest travel day of the year", "🍁 🦃 🌽",
    ),
    "Christmas Day": HolidayTheme(
        "🎄", "#0b3d2e", "#9b1b30", "#fdf6e3",
        "Very low commuting, unusual demand shape", "❄️ 🎄 🎁",
    ),
}


def theme_for(holiday_name: str) -> HolidayTheme:
    """Theme của một ngày lễ; rơi về theme trung tính nếu chưa khai báo riêng."""
    return HOLIDAY_THEMES.get(holiday_name, DEFAULT_THEME)
