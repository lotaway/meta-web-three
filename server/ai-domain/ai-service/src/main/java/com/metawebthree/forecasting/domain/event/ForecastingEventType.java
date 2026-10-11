package com.metawebthree.forecasting.domain.event;

public enum ForecastingEventType {
    FORECAST_CREATED,
    FORECAST_CONFIRMED,
    FORECAST_ADJUSTED,
    ACTUAL_SALES_RECORDED,
    MODEL_CREATED,
    MODEL_TRAINED,
    MODEL_DEPLOYED
}
