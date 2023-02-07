import schedule
import numpy as np
import datetime


def scheduler(times, function_to_call, args={}, iterate=False):
    """
    Inputs:
        times: array of ints, representing times in minutes from execution at which measurements should be made
        function_to_call: function or method must be pre-defined and be passed without parentheses
        args: if Iterate=False, dict of arguments that function_to_call takes
        iterate: if True, will iterate over the arguments held in args
    """
    # Scheduler set up takes too long to run function at time = 0
    if 0 in times:
        if iterate:
            function_to_call(args.pop(0))
        else:
            function_to_call(args)
    for t in times:
        time_hr = str((datetime.datetime.now() + datetime.timedelta(minutes=t)).hour)
        time_min = str((datetime.datetime.now() + datetime.timedelta(minutes=t)).minute)
        time_sec = str((datetime.datetime.now() + datetime.timedelta(minutes=t)).second)

        time_string = ''  # String to store the scheduled times

        for time_part in [time_hr, time_min, time_sec]:
            if len(time_part) == 1:
                # Add zero padding on left to ensure time is HH:MM:SS format when there are single digits
                time_part = time_part.zfill(2)
                time_string += (time_part + ':')
            else:
                time_string += (time_part + ':')

        # Remove the last ':' here instead of extra loop nesting
        time_string = time_string[:-1]

        # Add to schedule
        if iterate:
            schedule.every().day.at(time_string).do(lambda: function_to_call(args.pop(0)))
        else:
            schedule.every().day.at(time_string).do(function_to_call, **args)


class timingAccuracy():
    """
    Small test class to investigate the accuracy of this scheduler
    """

    def __init__(self, save_name='results.txt'):
        """
        Input:
            save_name (str): optional name of txt file to save data to
        """
        self.save_name = str(save_name)

    def collect(self):
        """
        This function is used to check how accurate the scheduler is by saving the microseconds of current time to file
        This function should be called over a range of times and can then be analysed using timingAccuracy.analyse()
        """
        file = open(self.save_name, 'a')
        full_dt = str(datetime.datetime.now()) + "\n"
        ms_dt = str(datetime.datetime.now().microsecond) + "\n"
        file.write(full_dt)
        file.write(ms_dt)
        file.close()

    def analyse(self, file_name=False):
        """
        Used to analyse the output files from timingAccuracy.collect()

        Input:
            file_name: (str) location of txt file. If blank, default 'results.txt' is used
        """
        if file_name is False:
            file_name = self.save_name

        file = open(str(file_name), 'r')
        microsecond_data = file.read()
        # Put data into list
        microsecond_list = microsecond_data.split("\n")
        # Close file
        file.close()
        # Only look at microseconds
        del microsecond_list[0::2]
        # Convert to ints
        data_into_list = [int(i) for i in microsecond_list]
        # Print standard deviation
        print("Standard deviation = 0." + str(np.std(data_into_list)) + "s")
