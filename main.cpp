/**
    Opencv example code: image features from moments (area, gravity center, perimeter...)
    Enrique Marin
    88enrique@gmail.com
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(){

    // Variables
    Mat src, src_gray;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Load an image
    src = imread("../Images/test.png");
    if( !src.data ){
        return -1;
    }

    // Remove noise by blurring with a Gaussian filter
    GaussianBlur( src, src, Size(3,3), 0.1, 0, BORDER_DEFAULT );

    // Convert the image to grayscale
    cvtColor(src, src_gray, CV_RGB2GRAY);

    // Find contoursin the image
    findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    //drawContours(src, contours, -1, cvScalar(0,0,255));

    printf("Number of contours: %d\n", (int)contours.size());

    // Get the moments from contours
    vector<Moments> mu(contours.size());
    for(int i = 0; i < (int)contours.size(); i++){
        mu[i] = moments(contours[i], false);
    }

    // Compute image features
    for(int i = 0; i < (int)contours.size(); i++){

        // Area, gravity center and perimeter from moments and contour length
        int area = mu[i].m00;
        int cx = mu[i].m10/mu[i].m00;
        int cy = mu[i].m01/mu[i].m00;
        int perimeter = arcLength(contours.at(i), true);

        // Orientation and axis from fitted ellipse
        RotatedRect r = fitEllipse(contours.at(i));
        double orientation = r.angle;
        double orientation_rads = orientation*3.1416/180;
        double majorAxis = r.size.height > r.size.width ? r.size.height : r.size.width;
        double minorAxis = r.size.height > r.size.width ? r.size.width : r.size.height;

        // Roundness, eccentricity, ratio and diameter from perimeter and axis
        double roundness = pow(perimeter, 2)/(2*3.1416*area);
        double eccentricity = sqrt(1-pow(minorAxis/majorAxis,2));
        double ratio = (minorAxis / majorAxis) * 100;
        double diameter = sqrt((4*area)/3.1416);

        // Print features
        printf("Area: %d\n", area);
        printf("Perimeter: %d\n", perimeter);
        printf("Major Axis: %.1f\n", majorAxis);
        printf("Minor Axis: %.1f\n", minorAxis);
        printf("Orientation: %.1f\n", orientation);
        printf("Roundness: %.1f\n", roundness);
        printf("Eccentricity: %.1f\n", eccentricity);
        printf("Ratio: %.1f\n", ratio);
        printf("Diameter: %.1f\n", diameter);
        printf("\n");

        // Draw fitted ellipse or rectangle
        //ellipse(src, r, cvScalar(0,255,0));
        rectangle(src, boundingRect(contours.at(i)), cvScalar(0,255,0));

        // Draw coordinate system
        //line(src, Point(cx-30, cy), Point(cx+30, cy), cvScalar(0,0,255));
        //line(src, Point(cx, cy-30), Point(cx, cy+30), cvScalar(0,0,255));

        // Draw orientation
        line(src, Point(cx, cy), Point(cx+30*cos(orientation_rads), cy+30*sin(orientation_rads)), cvScalar(255,0,0), 1);

        // Print orientation on image
        char tam [100];
        sprintf(tam, "Ori: %.0f", orientation);
        //putText(src, tam, Point(cx, cy), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(255));
    }

    // Show what you got
    imshow( "Original", src);

    waitKey(0);

    // Release memory
    src.release();
    src_gray.release();

    return 0;
}

