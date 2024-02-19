import plotly.express as px
from django.shortcuts import render



def plotly(request):

    # Bar Chart
    wide_df = px.data.medals_wide()

    fig_bar = px.bar(wide_df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input", height=300)

    # Scatter Chart
    df = px.data.iris()
    fig_scatter = px.scatter(df, x="sepal_width", y="sepal_length", color="species", height=300, hover_data=['petal_width'])

    # Line Chart
    df = px.data.gapminder().query("country in ['Canada', 'Botswana']")

    fig_line = px.line(df, x="lifeExp", y="gdpPercap", color="country", text="year", height=300)
    fig_line.update_traces(textposition="bottom right")

    # Pie Chart
    df = px.data.tips()
    fig_pie = px.pie(df, values='tip', names='day', height=300)


    bar_chart = fig_bar.to_html(full_html=False, include_plotlyjs=False)
    scatter_chart = fig_scatter.to_html(full_html=True, include_plotlyjs=False)
    line_chart = fig_line.to_html(full_html=False, include_plotlyjs=False)
    pie_chart = fig_pie.to_html(full_html=False, include_plotlyjs=False)

    return render(request, "plotly.html", {"bar_chart": bar_chart ,
                                           "scatter_chart" : scatter_chart, 
                                           "line_chart" : line_chart, 
                                           "pie_chart": pie_chart})