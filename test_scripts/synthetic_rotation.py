import subprocess

# Set the path to the sGLOH_opencv executable
executable_path = 'C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\cmake-build-debug-visual-studio\\sGLOH_opencv.exe'

# Set command line arguments
imageInputPath = "C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\src_img\\toucan.png"
folderPath = "C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\images"
args = [imageInputPath, folderPath]

# Open the output file in append mode
with open('output.txt', 'a') as f:
    # Run the program 10 times
    for i in range(10):
        # Run the executable and redirect its output to the file
        subprocess.run([executable_path] + args, stdout=f)

        # Write a newline character to separate the output from each run
        f.write('\n')
