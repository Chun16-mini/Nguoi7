
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Dự báo doanh số", layout="wide")

st.title("📈 DỰ BÁO DOANH SỐ")
st.write("Phân tích dữ liệu và dự báo doanh số bằng ARIMA/SARIMA.")

# 1. Tải dataset
st.sidebar.header("Dữ liệu")
file = st.sidebar.file_uploader(
    "Chọn dataset Excel hoặc CSV",
    type=["xlsx", "xls", "csv"]
)

if file is None:
    st.info("👈 Hãy tải dataset của nhóm ở thanh bên trái để bắt đầu.")
    st.stop()

# 2. Đọc dữ liệu
try:
    if file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, sep=None, engine="python")

    df.columns = df.columns.astype(str).str.strip()
    st.subheader("1. Xem dữ liệu")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("Tên các cột:", list(df.columns))

    # Chọn cột ngày và doanh số
    date_col = st.selectbox("Chọn cột thời gian", df.columns)
    sales_col = st.selectbox("Chọn cột doanh số", df.columns)

    data = df[[date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(
        data[date_col], dayfirst=True, errors="coerce"
    )
    data[sales_col] = pd.to_numeric(
        data[sales_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
    data = data.dropna().sort_values(date_col)
    data[date_col] = data[date_col].dt.to_period("M").dt.to_timestamp()

    series = data.groupby(date_col)[sales_col].sum().sort_index()
    series = series.asfreq("MS")

    if series.isna().any():
        st.warning("Có tháng thiếu dữ liệu; ứng dụng nội suy để chạy mô hình.")
        series = series.interpolate().ffill().bfill()

    if len(series) < 12:
        st.error("Cần ít nhất 12 tháng dữ liệu để dự báo.")
        st.stop()

    # 3. Biểu đồ lịch sử
    st.subheader("2. Doanh số lịch sử")
    st.line_chart(series)

    c1, c2 = st.columns(2)
    c1.metric("Số tháng", len(series))
    c2.metric("Doanh số trung bình", f"{series.mean():,.0f}")

    # 4. Thiết lập dự báo
    st.subheader("3. Dự báo doanh số")
    model_name = st.selectbox("Chọn mô hình", ["ARIMA", "SARIMA"])
    steps = st.slider("Số tháng dự báo", 1, 24, 12)

    if st.button("Chạy dự báo", type="primary"):
        with st.spinner("Đang tính toán..."):
            if model_name == "ARIMA":
                model = ARIMA(series, order=(1, 1, 1))
            else:
                model = SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

            fitted = model.fit()
            result = fitted.get_forecast(steps=steps)
            values = result.predicted_mean
            intervals = result.conf_int()

        future_dates = pd.date_range(
            start=series.index[-1] + pd.offsets.MonthBegin(1),
            periods=steps,
            freq="MS"
        )

        forecast = pd.DataFrame({
            "Tháng": future_dates,
            "Doanh số dự báo": values.to_numpy(),
            "Cận dưới 95%": intervals.iloc[:, 0].to_numpy(),
            "Cận trên 95%": intervals.iloc[:, 1].to_numpy()
        })

        st.subheader("4. Kết quả dự báo")
        st.dataframe(forecast, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series.index, series.values, label="Doanh số thực tế")
        ax.plot(
            future_dates, values.to_numpy(),
            marker="o", linestyle="--", label="Doanh số dự báo"
        )
        ax.fill_between(
            future_dates,
            intervals.iloc[:, 0].to_numpy(),
            intervals.iloc[:, 1].to_numpy(),
            alpha=0.2, label="Khoảng tin cậy 95%"
        )
        ax.set_title(f"Dự báo doanh số bằng {model_name}")
        ax.set_xlabel("Tháng")
        ax.set_ylabel("Doanh số")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Tải kết quả CSV",
            data=forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
