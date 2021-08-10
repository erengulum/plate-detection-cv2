# plate-detection-cv2
Car License Plate Detection

This study was carried out in three different stages. In the first stage,
reading, loading, converting to gray format, smoothing and edge detection
processes were carried out. The aim at this stage is to make our input as
smooth as possible before the Hough transform. In the next step, I applied
the Hough Transform method on the image we got from the previous stage.
For this, a Hough Space Accumulator was created first. Then, the image
was scanned pixel by pixel, and the transition was made from the (x,
y) plane to the Hough Space (rho, theta). Long and straight lines were
tried to be found by finding the peak points here. Since vehicle license
plates consist of two parallel lines that are perpendicular to each other,
similar structures have been implemented with different methods. This
rectangular structure was tried to be obtained with the help of different
methods. In the last step, the obtained lines were drawn on the original
image with the help of Python libraries. As a result, an attempt was made
to detect the plate.
