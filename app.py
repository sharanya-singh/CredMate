# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import io

# -----------------------------
# Utility Functions
# -----------------------------

@st.cache_data
def load_data(file):
    # if user uploads a file (BytesIO object)
    if file is not None:
        df = pd.read_csv(file)
    else:
        # fallback to default CSV in project folder
        df = pd.read_csv("cleaned_hpi.csv")
    
    # make sure Quarter column is datetime
    df['Quarter'] = pd.to_datetime(df['Quarter'])
    return df

def city_series(df, city):
    s = df[df["City"] == city].set_index("Date")["HPI"]
    s = s.groupby(s.index).mean()      # handle duplicate timestamps
    s = s.asfreq("QE")                 # quarterly frequency
    return s

def aic_search(y):
    """Brute-force SARIMA search by AIC (simplified)."""
    import warnings
    warnings.filterwarnings("ignore")
    
    best_aic, best_order, best_seasonal = 1e9, None, None
    
    for p in range(3):
        for d in range(2):
            for q in range(3):
                for sp in range(2):
                    for sd in range(2):
                        for sq in range(2):
                            try:
                                model = SARIMAX(
                                    y, 
                                    order=(p, d, q),
                                    seasonal_order=(sp, sd, sq, 4),  # Quarterly seasonality
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                res = model.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic, best_order, best_seasonal = res.aic, (p,d,q), (sp,sd,sq,4)
                            except:
                                continue
    return best_aic, best_order, best_seasonal

def forecast_sarima(y, steps=8):
    aic, order, seasonal = aic_search(y)
    model = SARIMAX(y, order=order, seasonal_order=seasonal)
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()

def forecast_prophet(y, steps=8):
    df = y.reset_index()
    df.columns = ["ds", "y"]

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="Q")
    forecast = m.predict(future)
    return forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🏠 House Price Index Forecasting Dashboard")

uploaded = st.file_uploader("Upload HPI CSV", type=["csv"])
if uploaded:
    df = load_data(uploaded)
    city = st.selectbox("Select City", df["City"].unique())

    if st.button("Run Forecasts"):
        series = city_series(df, city)

        # Run both forecasts
        sarima_mean, sarima_ci = forecast_sarima(series)
        prophet_forecast = forecast_prophet(series)

        # Combine forecasts for export
        forecast_df = pd.DataFrame({
            "Date": sarima_mean.index,
            "SARIMA_Forecast": sarima_mean.values,
            "Prophet_Forecast": prophet_forecast["yhat"].iloc[-len(sarima_mean):].values
        })

        # -----------------------------
        # Tabs for SARIMA, Prophet & Historical
        # -----------------------------
        tab1, tab2, tab3 = st.tabs(["📈 SARIMA Forecast", "🔮 Prophet Forecast", "📊 Historical Trends"])

        with tab1:
            st.subheader("SARIMA Forecast (AIC-tuned)")
            fig, ax = plt.subplots()
            ax.plot(series, label="Historical")
            ax.plot(sarima_mean, label="Forecast", color="orange")
            ax.fill_between(
                sarima_ci.index,
                sarima_ci.iloc[:, 0],
                sarima_ci.iloc[:, 1],
                color="orange",
                alpha=0.2,
            )
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.subheader("Prophet Forecast")
            fig2, ax2 = plt.subplots()
            ax2.plot(series, label="Historical")
            prophet_forecast[["yhat"]].plot(ax=ax2, color="green", label="Forecast")
            ax2.fill_between(
                prophet_forecast.index,
                prophet_forecast["yhat_lower"],
                prophet_forecast["yhat_upper"],
                color="green",
                alpha=0.2,
            )
            ax2.legend()
            st.pyplot(fig2)

        with tab3:
            st.subheader("📊 Historical Trends")
            fig3, ax3 = plt.subplots()
            ax3.plot(series, label="Historical HPI", color="blue")
            ax3.set_title(f"{city} - Historical HPI Trends")
            ax3.set_ylabel("HPI")
            ax3.legend()
            st.pyplot(fig3)

        # -----------------------------
        # Export CSV
        # -----------------------------
        st.subheader("Export Results")
        csv_buf = io.BytesIO()
        forecast_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download Forecasts as CSV",
            data=csv_buf.getvalue(),
            file_name=f"{city}_forecasts.csv",
            mime="text/csv"
        )
