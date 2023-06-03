import subprocess
import os
# Set the path to the sGLOH_opencv executable
executable_path = 'C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\cmake-build-debug-visual-studio\\sGLOH_opencv.exe'

# Set command line arguments
imageInputPath = "C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\src_img\\cabin.jpg"
folderPath = "C:\\Users\\Justin\\CLionProjects\\sGLOH_opencv\\images"
args = [imageInputPath, folderPath]

# Get a list of all the query images in the src_img folder to iterate over in the test loop
query_images = [f for f in os.listdir('src_img') if os.path.isfile(os.path.join('src_img', f))]
# Open the output file in write mode
with open('output.txt', 'w') as f:
    # Iterate over the query images
    for query_image in query_images:
        # Run the executable and redirect its output to the file
        subprocess.run([executable_path] + args, stdout=f)

        # Write a newline character to separate the output from each run
        f.write('\n')

# Run the clean script on the output file
subprocess.run(['python', 'clean.py', 'output.txt'])
