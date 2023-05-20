from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QFont
from PyQt5.QtCore import Qt, QUrl
import sys
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class UploadingData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None

    def download_data(self):
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period="max")
        self.data.dropna(inplace=True)
        self.data.reset_index(inplace=True)

    def save_data_to_csv(self, filename):
        self.data.to_csv(filename, index=False)

    def load_data(self):
        self.data = pd.read_csv(self.filepath)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Price Forecast")
        self.setWindowIcon(QIcon('1628696167_bitcoin_4.png'))
        self.setGeometry(100, 100, 1000, 1000)
        self.init_ui()

    def init_ui(self):
        # Установка фонового изображения
        palette = QPalette()
        palette.setBrush(self.backgroundRole(),
                         QBrush(QPixmap("news_16_09_2021_3_cr_market_1.jpg")))
        self.setPalette(palette)

        btc_data = UploadingData("BTC-USD")
        btc_data.download_data()
        btc_data.save_data_to_csv("BTC-USD.csv")
        bitcoin_data = btc_data.data

        input_sequence = bitcoin_data['Close'].values[-5:].reshape(1, 5, 1)

        self.num_days_label = QLabel("Введите количество дней для прогноза:")
        self.num_days_label.setStyleSheet("color: white; font-size: 46px; font-family: Rounds;")  # Изменение шрифта

        self.num_days_input = QLineEdit()
        self.num_days_input.setFixedWidth(400)
        self.num_days_input.setStyleSheet("font-size: 24px;")  # Изменение размера шрифта

        self.forecast_button = QPushButton("Прогнозировать")
        self.forecast_button.clicked.connect(self.show_forecast)
        self.forecast_button.setStyleSheet("font-size: 32px; font-family: Rounds;")  # Изменение размера и шрифта кнопки

        self.forecast_label = QLabel()
        self.forecast_label.setStyleSheet("color: black; font-size: 24px; font-family: Impact;")  # Изменение шрифта
        self.forecast_label.setWordWrap(True)
        self.forecast_label.setMinimumWidth(800)  # Изменение минимальной ширины метки прогноза

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)  # Изменение минимальной высоты
        scroll_area.setMinimumWidth(600)  # Изменение минимальной ширины
        scroll_area.setWidget(self.forecast_label)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.num_days_input)
        input_layout.addWidget(self.forecast_button)

        layout = QVBoxLayout()
        layout.addWidget(self.num_days_label, alignment=Qt.AlignTop)
        layout.addLayout(input_layout)
        layout.addWidget(scroll_area, alignment=Qt.AlignTop)

        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.plot_canvas)

        # Создание виджета для ссылки на GitHub
        github_label = QLabel()
        github_label.setOpenExternalLinks(True)
        github_label.setText(
            "<a href='https://github.com/MeikoFudo/CoursePaper'><img src='ad574c14aa17a899fd3abbf3cbbec62f.png' width='30' height='30'></a>")
        layout.addWidget(github_label, alignment=Qt.AlignBottom | Qt.AlignRight)
        layout.addWidget(github_label, alignment=Qt.AlignBottom | Qt.AlignRight)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Увеличение размера шрифта для поля ввода
        font = self.num_days_input.font()
        font.setPointSize(20)
        self.num_days_input.setFont(font)

    def show_forecast(self):
        model_path = 'C:\\Users\\mrhor\\Downloads\\my_model.h5'
        model = load_model(model_path)

        num_days = int(self.num_days_input.text())

        predicted_prices = []

        btc_data = UploadingData("BTC-USD")
        btc_data.download_data()
        btc_data.save_data_to_csv("BTC-USD.csv")
        bitcoin_data = btc_data.data

        input_sequence = bitcoin_data['Close'].values[-5:].reshape(1, 5, 1)

        for _ in range(num_days):
            predicted_price = model.predict(input_sequence)
            predicted_prices.append(predicted_price[0][0])
            input_sequence = np.append(input_sequence[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

        forecast_text = ""
        for i, price in enumerate(predicted_prices):
            forecast_text += f"Прогнозирования для {i + 1} дня: {price}\n"

        self.forecast_label.setText(forecast_text.strip())

        actual_prices = bitcoin_data['Close'].values
        self.plot_price_forecast(actual_prices, predicted_prices)

    def plot_price_forecast(self, actual_prices, forecast_prices):
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.plot(actual_prices, label='Фактическая цена', color='blue')
        self.plot_canvas.axes.plot(range(len(actual_prices), len(actual_prices) + len(forecast_prices)),
                                   forecast_prices,
                                   label='Прогнозируемая цена', color='red')
        self.plot_canvas.axes.set_xlabel('День')
        self.plot_canvas.axes.set_ylabel('Цена')
        self.plot_canvas.axes.grid(True)
        self.plot_canvas.axes.set_title('Прогноз цены')
        self.plot_canvas.axes.legend()
        self.plot_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
