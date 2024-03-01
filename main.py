import csv
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.linear_model import LinearRegression


def retention_time_to_seconds(time_str):
    m, s = map(int, time_str.split(':'))
    return m * 60 + s


def seconds_to_retention_time(seconds):
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


class Session:
    def __init__(self, timestamp: str, retention_times: List[str]):
        self.timestamp = datetime.strptime(timestamp, "%H:%M,%d/%m/%Y")
        self.retention_times = [retention_time_to_seconds(t) for t in retention_times]

    def best_retention_time(self):
        return max(self.retention_times)

    def average_retention_time(self):
        return sum(self.retention_times) / len(self.retention_times)

    def total_rounds(self):
        return len(self.retention_times)


class SessionLog:
    def __init__(self, filename: str):
        self.filename = filename
        self.sessions = self.load_sessions()

    def load_sessions(self):
        seshs = []
        try:
            with open(self.filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    # skip rows that don't contain valid datetime data
                    if row and not row[0].startswith('hr:') and not row[1].endswith('yyyy'):
                        try:
                            session = Session(row[0] + ',' + row[1], row[2:])
                            seshs.append(session)
                        except ValueError as e:
                            print(f"Skipping row due to error: {e}")
        except FileNotFoundError:
            print(f"File '{self.filename}' not found")

        return seshs

    def add_session(self, session: Session):
        self.sessions.append(session)
        with open(self.filename, 'a') as file:
            csv_writer = csv.writer(file)
            row = [session.timestamp.strftime("%H:%M,%d/%m/%Y")] + [seconds_to_retention_time(t) for t in session.retention_times]
            csv_writer.writerow(row)


class DataAnalysis:
    def __init__(self, sessions):
        self.sessions = sessions

    def overall_average_retention_time(self):
        total_time = sum([sum(session.retention_times) for session in self.sessions])
        total_rounds = sum([len(session.retention_times) for session in self.sessions])
        return round(total_time / total_rounds) if total_rounds else 0


class DataVisualization:
    def __init__(self, sessions, background_color='grey'):
        self.sessions = sessions
        self.background_color = background_color
        self.dates_all, self.all_retention_times, self.best_retention_times, self.average_retention_times, self.session_dates = self.prepare_data()
        self.fig = None
        self.ax = None

    def prepare_data(self):
        dates_all = []
        all_retention_times = []
        best_retention_times = []
        average_retention_times = []
        session_dates = []

        for session in self.sessions:
            session_date = session.timestamp
            session_dates.append(session_date)
            retention_times = session.retention_times
            best_time = session.best_retention_time()
            average_time = session.average_retention_time()

            dates_all.extend([session_date] * len(retention_times))
            all_retention_times.extend(retention_times)
            best_retention_times.append(best_time)
            average_retention_times.append(average_time)

        return dates_all, all_retention_times, best_retention_times, average_retention_times, session_dates

    def initialize_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')  # Rotate for better legibility
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Retention Time (seconds)')
        self.ax.set_title('Breathwork Session Analysis')

    def plot_all_sessions(self):
        dates_formatted = mdates.date2num(self.dates_all)
        self.ax.scatter(dates_formatted, self.all_retention_times, label='All Retention Times', alpha=0.5)

    def plot_best_times(self):
        unique_dates_formatted = mdates.date2num(self.session_dates)
        self.ax.scatter(unique_dates_formatted, self.best_retention_times, color='yellow', label='Best Retention Times',
                        edgecolor='black')

    def plot_session_averages(self):
        unique_dates_formatted = mdates.date2num(self.session_dates)
        self.ax.scatter(unique_dates_formatted, self.average_retention_times, color='purple', label='Session Averages',
                        edgecolor='black', zorder=5)

    def add_data_regression(self):
        self.add_regression_line(self.dates_all, self.all_retention_times, 'Trend', '#1f77b4')

    def add_best_times_regression(self):
        self.add_regression_line(self.session_dates, self.best_retention_times, 'Best Times Trend', 'Gold')

    def add_regression_line(self, x_dates, y, label, color):
        x_dates_formatted = mdates.date2num(x_dates)
        x = np.array(x_dates_formatted).reshape(-1, 1)
        y = np.array(y)

        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)

        self.ax.plot(x_dates_formatted, y_pred, label=label, color=color, linewidth=2, alpha=0.75)

    def finalize_plot(self, overall_average):
        self.fig.autofmt_xdate()
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), facecolor=self.background_color, framealpha=1)
        self.fig.subplots_adjust(right=0.8)
        self.ax.text(1.01, 0.7, f'Average Retention: {overall_average}s',transform=self.ax.transAxes, verticalalignment='top')
        self.fig.show()

    def save_plot(self, filename, bbox_inches='tight'):
        self.fig.savefig(filename, bbox_inches=bbox_inches, facecolor=self.fig.get_facecolor())

def main():
    # Load sessions
    session_log = SessionLog('sessions.txt')
    sessions = session_log.sessions

    # Analyze data
    data_analysis = DataAnalysis(sessions)
    overall_average = data_analysis.overall_average_retention_time()
    print(f"Overall Average Retention Time: {overall_average} seconds")

    # Visualize data
    data_visualization = DataVisualization(sessions)
    data_visualization.initialize_plot()
    data_visualization.plot_all_sessions()
    data_visualization.plot_best_times()
    data_visualization.plot_session_averages()
    data_visualization.add_data_regression()
    data_visualization.add_best_times_regression()
    data_visualization.finalize_plot(overall_average)

if __name__ == "__main__":
    main()

