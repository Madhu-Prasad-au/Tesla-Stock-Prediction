import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

def dataset_summary(df):

    summary = {

        "Rows": df.shape[0],

        "Columns": df.shape[1],

        "Missing Values":
        df.isnull().sum().sum(),

        "Duplicate Rows":
        df.duplicated().sum(),

        "Start Date":
        "29/6/2010",

        "End Date":
        "3/2/2020"
    }

    return summary

def show_summary_cards(df):

    summary = dataset_summary(df)

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Rows",
        summary["Rows"]
    )

    c2.metric(
        "Columns",
        summary["Columns"]
    )

    c3.metric(
        "Missing Values",
        summary["Missing Values"]
    )

    c4,c5,c6 = st.columns(3)

    c4.metric(
        "Duplicates",
        summary["Duplicate Rows"]
    )

    c5.metric(
        "Start Date",
        (summary["Start Date"])
    )

    c6.metric(
        "End Date",
        (summary["End Date"])
    )



def show_dataset_preview(df):

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

def show_missing_values(df):

    st.subheader(
        "Missing Values"
    )

    missing = pd.DataFrame({

        "Column":
        df.columns,

        "Missing Count":
        df.isnull().sum().values

    })

    st.dataframe(
        missing,
        use_container_width=True
    )


def show_statistics(df):

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

def close_price_chart(df):

    fig = px.line(

        df,

        x="Date",

        y="Close",

        title="Tesla Closing Price Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def volume_chart(df):

    fig = px.line(

        df,

        x="Date",

        y="Volume",

        title="Trading Volume"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def moving_average_chart(df):

    data = df.copy()

    data["MA10"] = (
        data["Close"]
        .rolling(10)
        .mean()
    )

    data["MA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=data["Date"],

            y=data["Close"],

            name="Close"
        )
    )

    fig.add_trace(

        go.Scatter(

            x=data["Date"],

            y=data["MA10"],

            name="MA10"
        )
    )

    fig.add_trace(

        go.Scatter(

            x=data["Date"],

            y=data["MA50"],

            name="MA50"
        )
    )

    fig.update_layout(

        title="Moving Average Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def close_distribution(df):

    fig = px.histogram(

        df,

        x="Close",

        nbins=50,

        title="Close Price Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def close_boxplot(df):

    fig = px.box(

        df,

        y="Close",

        title="Close Price Outliers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig = px.imshow(

        corr,

        text_auto=True,

        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def candlestick_chart(df):

    fig = go.Figure(

        data=[

            go.Candlestick(

                x=df["Date"],

                open=df["Open"],

                high=df["High"],

                low=df["Low"],

                close=df["Close"]

            )

        ]
    )

    fig.update_layout(

        title="Tesla Candlestick Chart"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def run_complete_eda(df):

    show_summary_cards(df)

    show_dataset_preview(df)

    show_missing_values(df)

    show_statistics(df)

    close_price_chart(df)

    volume_chart(df)

    moving_average_chart(df)

    close_distribution(df)

    close_boxplot(df)

    correlation_heatmap(df)

    candlestick_chart(df)

    
