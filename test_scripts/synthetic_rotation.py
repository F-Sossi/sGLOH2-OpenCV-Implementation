import subprocess

# Set the path to the sGLOH_opencv executable
executable_path = 'C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\cmake-build-debug-visual-studio\\sGLOH_opencv.exe'

# Set command line arguments
imageInputPath = "C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\src_img\\toucan.png"
folderPath = "C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\images"
args = [imageInputPath, folderPath]

# Run the executable and capture its output
result = subprocess.run([executable_path] + args, stdout=subprocess.PIPE)

# Print the output
print(result.stdout.decode('utf-8'))
