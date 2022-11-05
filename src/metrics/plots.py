import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def effect_plot_ptl(res, left_border, right_border, title, output_path, name):
    figure = plt.gca()
    plt.figure(figsize=(20, 10))
    plt.hist(res, bins=200, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(np.mean(res), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(left_border, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(right_border, color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    plt.text(np.mean(res) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}% '.format(np.mean(res)))
    plt.text(left_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(left_border))
    plt.text(right_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(right_border))
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.title(title)
    plt.savefig(output_path + '/' + name + '_conf_level.png')

def effect_plot_px(res, left_border, right_border, title, name):
    #fig = px.histogram(res, nbins=200, color_discrete_sequence=['lightseagreen'], title=title)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3])
    fig.add_trace(go.Histogram(
            x=res,
            name=title,
            nbinsx=200,
            marker_color='lightseagreen'
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=np.mean(res),
        line_width=1, line_dash="dash", line_color="red",
        annotation_text='Mean: {:.2f}% '.format(np.mean(res)), annotation_position="top right",
        row=1, col=1
    )
    fig.add_vline(
        x=left_border,
        line_width=1, line_dash="dash", line_color="red",
        annotation_text='{:.2f}% '.format(left_border), annotation_position="top right",
        row=1, col=1
    )
    fig.add_vline(
        x=right_border,
        line_width=1, line_dash="dash", line_color="red",
        annotation_text='{:.2f}% '.format(right_border), annotation_position="top right",
        row=1, col=1
    )

    fig.add_trace(
        go.Box(
            y=res,
            name=title,
            marker_color='lightseagreen'
        ), row=1, col=2
    )

    fig.update_layout(title_text=title, showlegend=False)
    save_html(fig, name + '_conf_level')


def save_html(
        fig,
        plot_name,
        height_c=500,
        width_tag='small'
):
    folder = '../plots/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    if width_tag.lower() == 'small' or width_tag.lower() == 's':
        width_c = 700
    elif width_tag.lower() == 'medium' or width_tag.lower() == 'm':
        width_c = 900
    elif width_tag.lower() == 'big' or width_tag.lower() == 'lagge' or width_tag.lower() == 'l':
        width_c = 1100
    else:
        print('Incorrect width_tag! Set to default (700px, small)')
        width_c = 700

    # change size for confluence
    fig.update_layout(
        autosize=False,
        width=width_c,
        height=height_c
    )

    # convert plot to JSON
    fig_json = fig.to_json()

    # a simple HTML template
    template = """<html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id='divPlotly'></div>
        <script>
            var plotly_data = {}
            Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
        </script>
    </body>

    </html>"""

    # write the JSON to the HTML template
    with open(folder + plot_name + '.html', 'w') as f:
        f.write(template.format(fig_json))