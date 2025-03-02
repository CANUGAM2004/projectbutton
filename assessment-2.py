import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_circles(image_path, min_radius=10, max_radius=100, param1=50, param2=30):
    img = cv2.imread("detectedimage.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,          
        minDist=20,     
        param1=param1,  
        param2=param2,  
        minRadius=min_radius,
        maxRadius=max_radius
    )
    output_image = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return circles, output_image

def visualize_results(original_image, output_image, circles=None):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Circles: {0 if circles is None else len(circles[0])}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "detectedimage.jpeg"
    original = cv2.imread(image_path)
    circles, output_image = detect_circles(
        image_path, 
        min_radius=10,
        max_radius=100,
        param1=50,
        param2=30
    )
    if circles is not None:
        print(f"Found {len(circles[0])} circles")
        for i, circle in enumerate(circles[0]):
            print(f"Circle {i+1}: Center=({circle[0]}, {circle[1]}), Radius={circle[2]}")
    else:
        print("No circles found")
    
    visualize_results(original, output_image, circles)