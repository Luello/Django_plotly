import plotly.express as px
from django.shortcuts import render
from .analyse import first_graph,fig,heat,boxplotsO,income ,elbow,PCA2D,PCA3D,figRFM

def page_2(request):
    return render(request, 'page_2.html')



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

    first_graph_html = first_graph.to_html(full_html=False, include_plotlyjs=False)
    fig2_html= fig.to_html(full_html=False, include_plotlyjs=False)
    #fig3_html= fig3.to_html(full_html=False, include_plotlyjs=False)
    heat_html= heat.to_html(full_html=False, include_plotlyjs=False)
    boxplotsO_html= boxplotsO.to_html(full_html=False, include_plotlyjs=False)
    income_html= income.to_html(full_html=False, include_plotlyjs=False)
    elbow_html= elbow.to_html(full_html=False, include_plotlyjs=False)
    PCA2D_html= PCA2D.to_html(full_html=False, include_plotlyjs=False)
    PCA3D_html= PCA3D.to_html(full_html=False, include_plotlyjs=False)
    figRFM_html=figRFM.to_html(full_html=False, include_plotlyjs=False)
    return render(request, "plotly.html", {"bar_chart": bar_chart ,
                                           "scatter_chart" : scatter_chart, 
                                           "line_chart" : line_chart, 
                                           "pie_chart": pie_chart,
                                           "first_graph":first_graph_html,
                                           "fig2_html" : fig2_html,
                                           "income_html": income_html,
                                           "heat":heat_html  ,
                                           "boxplotsO":boxplotsO_html,
                                           "elbow": elbow_html,
                                           "PCA2D":PCA2D_html,
                                           "PCA3D":PCA3D_html,
                                           "figRFM":figRFM_html
                                           
                                           })