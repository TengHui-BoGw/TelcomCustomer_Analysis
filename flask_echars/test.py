from flask import Flask, render_template
from pyecharts import options as opts
from pyecharts.charts import Line, Scatter

app = Flask(__name__)

@app.route('/')
def index():
    x_data = ['A', 'B', 'C', 'D', 'E']
    y_data1 = [10, 20, 30, 40, 50]
    y_data2 = [20, 30, 40, 50, 60]

    # 创建折线图
    line_chart = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis('折线图1', y_data1)
        .add_yaxis('折线图2', y_data2)
        .set_global_opts(title_opts=opts.TitleOpts(title='折线图'))
    )

    # 创建散点图
    scatter_chart = (
        Scatter()
        .add_xaxis(x_data)
        .add_yaxis('散点图', y_data1)
        .set_global_opts(title_opts=opts.TitleOpts(title='散点图'))
    )

    # 渲染模板并返回
    return render_template('index.html', line_chart=line_chart.render_embed(), scatter_chart=scatter_chart.render_embed(), host='http://localhost:5000/')

if __name__ == '__main__':
    app.run(debug=True)
