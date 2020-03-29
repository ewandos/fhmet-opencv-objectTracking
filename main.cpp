#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main( int argc, char** argv ) {
    
    // Variables
    Mat imgMat;
    Mat templMat;
    Mat korrMat;
    String imageWindow = "Source Image";
    String korrWindow = "Korrelation window";
    int matchingMethod = TM_CCOEFF_NORMED;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    double minVal;
    double maxVal;
    
    // Load Sources
    templMat = imread("helm.png");
    VideoCapture cap("sprung.mp4");
    
    if(!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    
    // Create windows
    namedWindow( imageWindow, WINDOW_AUTOSIZE );
    namedWindow( korrWindow, WINDOW_AUTOSIZE );
    
    while (1) {
        
        cap >> imgMat;
        
        if (imgMat.empty() || templMat.empty())
              break;

        // Source image to display
        Mat img_display;
        imgMat.copyTo( img_display );
        
        // Create the result matrix
        int result_cols =  imgMat.cols - templMat.cols + 1;
        int result_rows = imgMat.rows - templMat.rows + 1;
        korrMat.create( result_cols, result_rows, CV_32FC1 );
        
        // Do the Matching with normalized matching method (TM_CCORR_NORMED)
        matchTemplate( imgMat, templMat, korrMat, matchingMethod);
        
        
        // find k matches of given template in image
        // get current minimum and maximum
        minMaxLoc( korrMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        
        // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
        if( matchingMethod  == TM_SQDIFF || matchingMethod == TM_SQDIFF_NORMED ) {
            matchLoc = minLoc;
            // fill the found minima with a rectangle on the result Mat to avoid finding the same minima multiple times
            rectangle( korrMat, matchLoc, Point( matchLoc.x + templMat.cols + 20, matchLoc.y + templMat.rows + 20), Scalar::all(255), FILLED, 8, 0 );
        } else {
            matchLoc = maxLoc;
            cout << maxVal << endl;
            // fill the found maxima with a rectangle on the result Mat to avoid finding the same maxima multiple times
            rectangle( korrMat, matchLoc, Point( matchLoc.x + templMat.cols + 20, matchLoc.y + templMat.rows + 20), Scalar::all(0), FILLED, 8, 0 );
        }
        
        // mark the found match on the img Mat
        rectangle( img_display, matchLoc, Point( matchLoc.x + templMat.cols , matchLoc.y + templMat.rows ), Scalar::all(0), 2, 8, 0 );
        
        // Show both images
        imshow( imageWindow, img_display );
        imshow( korrWindow, korrMat );
        
        waitKey(1); // change to 0 For skipping Frame by Frame
        
    }
    
    waitKey(0);
    cap.release();
    
    return 0;
}
