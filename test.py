import pandas as pd
from lightweight_charts import Chart

if __name__ == '__main__':
    df = pd.read_csv('BTC1D.csv')
    df.reset_index()
    df.columns=df.columns.str.lower()
    chart = Chart()
    chart.set(df)
    chart.layout(background_color='#090008', text_color='#FFFFFF', font_size=16, font_family='Helvetica')

    chart.candle_style(up_color='#00ff55', down_color='#ed4807', border_up_color='#FFFFFF', border_down_color='#FFFFFF',
                       wick_up_color='#FFFFFF', wick_down_color='#FFFFFF')

    chart.volume_config(up_color='#00ff55', down_color='#ed4807')

    chart.watermark('1D', color='rgba(180, 180, 240, 0.7)')  
    chart.show(block=True)