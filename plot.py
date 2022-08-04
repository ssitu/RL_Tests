import random
import time

from matplotlib import pyplot as plt
import matplotlib

import multiprocessing as mp


class Plot(mp.Process):
    """
    Class to manage the plot
    If the Plot.pause() is called, and the main process ends before calling Plot.stop() or Plot.unpause(),
    the plot process will continue to run until the figure is closed manually,
    and the main process will hang until then.
    """

    def __init__(self, title, x_label, y_label, moving_avg_len=100):
        super(Plot, self).__init__()

        # Initialize multiprocessing variables
        # The list that is shared between the main process and the plot process
        self.shared_list = mp.Manager().list()
        # The pause event to pause the plot loop
        self.pause_event = mp.Event()
        # The stop event to stop the plot loop
        self.stop_event = mp.Event()
        # The event to clear the local data
        self.clear_event = mp.Event()

        # Initialize parameters
        # The moving average length
        self.moving_avg_len = moving_avg_len
        # Plot labels
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

        # Initialize variables that are given values in init_plot_system()
        self.fig = None
        self.ax = None
        self.data = None
        self.moving_averages = None
        self.update_interval = None
        self.moving_avg = None

    def init_plot_system(self):
        # Set up the plot
        # Obtain the figure and axes
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # Simplify for faster rendering
        matplotlib.rcParams['path.simplify'] = True
        matplotlib.rcParams['path.simplify_threshold'] = 1.0
        # The data to plot
        self.data = []
        # The moving average of the data
        self.moving_averages = []
        # The current moving average
        self.moving_avg = 0
        # The number of seconds to wait between updates
        self.update_interval = 0.01

    def run(self):
        """
        Run the plot process
        """
        self.init_plot_system()
        self.plot_loop()

    def format_plot(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

    def add_data(self, val):
        """
        Add data to the shared list
        :param val: The value to add to the list
        :return:
        """
        self.shared_list.append(val)

    def update_local_data(self, val):
        """
        Add data to the plot
        :param val: The value to add to the plot
        """
        # Add the new value to the data
        self.data.append(val)
        #
        # Moving average
        #
        # 2 cases:
        # 1. The number of data points is greater than the moving average length
        # 2. The number of data points is less than the moving average length
        # 1)
        # Multiply the current average by the moving average length,
        # subtract the oldest value, and add the new value, then divide by the moving average length
        # to get the new average.
        # 2)
        # Use the size of the moving averages as the moving average length.
        # Using the same formula to get the new average, except the oldest value is not subtracted.
        moving_avg_size = len(self.moving_averages)
        if moving_avg_size < self.moving_avg_len:
            self.moving_avg = (self.moving_avg * moving_avg_size + val) / (moving_avg_size + 1)
        else:
            old_val = self.data[-self.moving_avg_len]
            self.moving_avg = (self.moving_avg * self.moving_avg_len - old_val + val) / self.moving_avg_len
        # Add the new average to the moving average list
        self.moving_averages.append(self.moving_avg)

    def plot_loop(self):
        # Catch the EOFError to prevent the plotting process from crashing.
        try:
            # Setup matplotlib
            # Settings to Simplify the rendering of the plot.
            matplotlib.rcParams['path.simplify'] = True
            matplotlib.rcParams['path.simplify_threshold'] = 1.0
            # Start the plot loop
            while True:
                #
                # Respond to events
                #
                # Stay in the loop until the pause event is not set
                while self.pause_event.is_set() and not self.stop_event.is_set() and plt.get_fignums():
                    plt.pause(self.update_interval)
                # Exit the loop if the stop event is set
                if self.stop_event.is_set():
                    print("Plot stopped")
                    break
                # Clear the data if the clear event is set
                if self.clear_event.is_set():
                    self.__clear()

                #
                # Preconditions
                #
                # Data may be empty
                if self.shared_list is None:
                    continue
                # Only update if there is more data added to the shared list than there is in the local data list
                shared_len = len(self.shared_list)
                if shared_len <= len(self.data):
                    continue

                #
                # Plotting
                #
                # Add the new data to the lists
                # Loop through the new data
                for i in range(len(self.data), shared_len):
                    # Add the new data
                    self.update_local_data(self.shared_list[i])
                # Update the plot
                self.ax.cla()
                self.format_plot()
                self.ax.plot(self.data)
                self.ax.plot(self.moving_averages)
                plt.pause(self.update_interval)
        except (BrokenPipeError, EOFError) as e:
            print(e)
            print("The plotting process reached an error. The main process may have ended.")

    def pause(self):
        """
        Pause the plot loop
        """
        self.pause_event.set()

    def unpause(self):
        """
        Unpause the plot loop
        """
        self.pause_event.clear()

    def stop(self):
        """
        Stop the plot loop
        """
        self.stop_event.set()

    def clear_data(self):
        self.clear_event.set()

    def __clear(self):
        # Clear the data
        self.data = []
        # Clear the moving averages
        self.moving_averages = []
        # Clear the moving average
        self.moving_avg = 0
        # Clear the shared list
        self.shared_list[:] = []
        self.clear_event.clear()  # Clear the clear event

    def save(self, filename):
        """
        Save the plot to a file
        :param filename: The filename to save the plot to
        """
        # Since this is called from the Plot object in the main process, the Plot object does not have the same figure.
        # So the figure must be generated again.
        self.init_plot_system()
        self.ax.cla()
        self.format_plot()
        self.ax.plot(self.shared_list)
        # Use the data from the shared list to calculate the moving averages
        for i in range(len(self.shared_list)):
            self.update_local_data(self.shared_list[i])
        self.ax.plot(self.moving_averages)
        plt.savefig(filename)


if __name__ == '__main__':
    plot = Plot("TITLE", "X-AXIS", "Y-AXIS", moving_avg_len=100)
    plot.start()
    # Start the loop that generates random numbers for the y values, and increasing values for the x values
    for _ in range(1000):
        # Generate a random number
        y = random.random()
        # Add the data to the list
        plot.add_data(y)
        # Wait for a bit
        time.sleep(0.0001)
        if _ == 500:
            plot.clear_data()
