#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>
#include <fstream>
using namespace std;
using namespace std::chrono;
using namespace cv;

int* cpuOperation(Mat image, float BGRweights[], int size);
Mat copyValues(Mat image, int grayArray[]);

int main( int argc, char** argv)
{

	auto start = high_resolution_clock::now();

	Mat image;
	Mat grayImage;	
	
	image = imread( argv[1], CV_LOAD_IMAGE_COLOR); 

	grayImage = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	cv::FileStorage file("Gray.xml", cv::FileStorage::WRITE);
	
	file << "Gray" << grayImage;


	if(argc !=2 || !image.data)
	{
		printf( "No image data \n" );
		return -1;
	}

	//adds up pixels
	int totalPixels = image.total(); 
		
	float BGRweights[3] = {0.114, 0.587, 0.299}; 

	int* grayArray;
	grayArray = (int *) malloc(totalPixels*sizeof(int));
	
	grayArray=cpuOperation(image, BGRweights, totalPixels);

	image=copyValues(image, grayArray);
	
	Mat splitChannels[3];
	split(image, splitChannels);

	//imshow( "Display Image", splitChannels[0]);
	//waitKey(0);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

	return 0;
}

Mat copyValues(Mat image, int grayArray[]){
        int counter = 0;	
	for(int row=0;row<image.rows;row++)
		{
			for(int col=0;col<image.cols;col++)
			{
				image.at<cv::Vec3b>(row, col)[0] = grayArray[counter];
				counter++;
			}
		}

	return image;

}

int* cpuOperation(Mat image, float BGRweights[], int size){

	int counter = 0;

	int* grayArray;
	grayArray = (int *) malloc(size*sizeof(int));
	
	for(int row=0;row<image.rows;row++)
	{
		for(int col=0;col<image.cols;col++)
		{
			int B = (int) image.at<Vec3b>(row, col)[0] * BGRweights[0];
			int G = (int) image.at<Vec3b>(row, col)[1] * BGRweights[1]; 
			int R = (int) image.at<Vec3b>(row, col)[2] * BGRweights[2];
			grayArray[counter]=B+G+R;	
			counter++;
		}
	}

        return grayArray;
}

