import os
import subprocess

# Define the directory where your program and data are located
working_directory = "/home/yiming/projects/duckdb/cmake-build-release"

# Change the current working directory to the program directory
os.chdir(working_directory)

# Name of the program (assumes it's in the working_directory)
program_name = "reservoir_main"

# Full path to the executable
program_path = os.path.join(working_directory, program_name)

# List of total number of threads
total_threads_list = [96]

# List of reservoir thread counts
thread_counts = [1, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]

# NUMA settings
cpunode = 1
membind = 1

# Iterate over total threads and thread counts
for total_threads in total_threads_list:
    for left_threads in thread_counts:
        print(
            f"Running program with total {total_threads} threads and {left_threads} reservoir thread(s) on CPU node {cpunode} and memory node {membind}...")

        # Construct the base command
        command = ["numactl"]

        # Add NUMA options only if cpunode and membind are not -1
        if cpunode != -1:
            command.append(f"--cpunodebind={cpunode}")
        if membind != -1:
            command.append(f"--membind={membind}")

        # Add program path and arguments
        command.extend([
            program_path,
            str(total_threads),  # Pass total threads as an argument
            str(left_threads)  # Pass the number of left_threads as an argument
        ])

        # Print the constructed command for debugging
        # print(f"Executing command: {' '.join(command)}")

        # Run the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        # Print the program's output
        if result.stderr:
            print(f"Total {total_threads} threads and {left_threads} reservoir thread(s):\n{result.stderr}")
