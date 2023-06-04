import sys
import os

# Clean the output from test scripts
# Open the file passed in as the first argument
with open(sys.argv[1], 'r+') as f:
    # If the file is empty, exit
    if os.stat(sys.argv[1]).st_size == 0:
        print('File is empty')
        exit()
    # Read the file into a list of lines
    lines = f.readlines()
    # Process the output
    # Remove any lines that start with [
    lines = [line for line in lines if not line.startswith('[')]
    # Remove any lines "Finished processing images"
    lines = [line for line in lines if not line.startswith('Finished processing images')]
    # Remove any lines that start with "Processed image"
    lines = [line for line in lines if not line.startswith('Processed image')]
    # C: leaving only the file name and the match score
    lines = [line.replace('C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\images\\', '') for line in lines]
    # If it is the last line in a test add an empty line
    lines = [line if not line.startswith('SIFT descriptor took') else line + '\n' for line in lines]
    # Close the file in read mode and open in write mode
    f.close()
    f = open(sys.argv[1], 'w')
    # Write the processed output to the file
    f.writelines(lines)

