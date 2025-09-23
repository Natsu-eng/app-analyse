
import plotly.graph_objects as go

DATALAB_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif"),
        title_font=dict(family="sans-serif", size=22),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
    )
)
