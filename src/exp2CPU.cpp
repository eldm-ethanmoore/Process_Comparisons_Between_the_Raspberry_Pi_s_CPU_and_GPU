#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <sstream>
#include <chrono>
#include <string>
using namespace cv;
using namespace std;
using namespace std::chrono;

float averageIntervalTime = 0;
int intervalCounter = 0;

int *frameOne;
int *frameTwo;
int *frameThreeDiff;

int main() {

	Mat myImage;
	namedWindow("video player");
	VideoCapture cap(0);

	if(!cap.isOpened())
	{
		cout << "cannot open camera" << endl;
		system("pause");
		return -1;
	}
	
	int stepNumber = 0;

	auto totalTimeEndStart = std::chrono::system_clock::now();
	while(true)
	{


		std::stringstream streamOfWindowName;
		streamOfWindowName << "Average speed of frames every 5 seconds: " << averageIntervalTime; 
		std::string strOfWindowName = streamOfWindowName.str();

		cap >> myImage;
		if(myImage.empty())
		{
			break;
		}
	
		Mat grayMyImage;
		cvtColor(myImage, grayMyImage, COLOR_BGR2GRAY);

		int counter = 0;
		
		if(stepNumber==0)
		{
			frameOne = (int*) malloc(grayMyImage.total()*sizeof(int));	

			for(int row=0;row<grayMyImage.rows;row++)
			{
				for(int col=0;col<grayMyImage.cols;col++)
				{

					frameOne[counter] = grayMyImage.at<Vec3b>(row, col)[0];
					counter++;
				}
			}
			stepNumber = 1;

			imshow("video player", grayMyImage);
			char c = (char)waitKey(25);
			if(c == 27)
			{
				break;
			}
			continue;
		}

		if(stepNumber==1)
		{

			frameTwo = (int*) malloc(grayMyImage.total()*sizeof(int));	
			for(int row=0;row<grayMyImage.rows;row++)
			{
				for(int col=0;col<grayMyImage.cols;col++)
				{

					frameTwo[counter] = grayMyImage.at<Vec3b>(row, col)[0];
					counter++;
				}
			}

			counter = 0;


			

			frameThreeDiff = (int*) malloc(grayMyImage.total()*sizeof(int));	

                        auto startComputationInterval = high_resolution_clock::now();
			for(int row=0;row<grayMyImage.rows;row++)
			{
				for(int col=0;col<grayMyImage.cols;col++)
				{
					
					int f1px = frameOne[counter];
					int f2px = frameTwo[counter];
					int diffPixel = abs(f1px-f2px);
					frameThreeDiff[counter]=diffPixel;	
					counter++;
					
					if(diffPixel > 70)
					{
						grayMyImage.at<Vec3b>(row, col)[0] = 255;
					}
					
				}
			}
			auto stopComputationInterval = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(stopComputationInterval - startComputationInterval);

			free(frameOne);
			free(frameTwo);
			counter = 0;

			for(int row=0;row<grayMyImage.rows;row++)
			{
				for(int col=0;col<grayMyImage.cols;col++)
				{
					
					if(frameThreeDiff[counter] > 70)
					{
						grayMyImage.at<Vec3b>(row, col)[0] = 255;
					}
					counter++;
					
				}
			}

			stepNumber = 0;
			free(frameThreeDiff);


			fstream OUTPUT;
			std::stringstream ss;
			ss << "CPUInterval" << intervalCounter << ".txt";
			std::string s = ss.str();
			OUTPUT.open(s, ios::app);
			OUTPUT << duration.count() << endl;

			OUTPUT.close();
			auto totalTimeEnd = std::chrono::system_clock::now();
			auto elapsed = totalTimeEnd - totalTimeEndStart;
			if(elapsed.count()  >= 9000000000 )
			{
				OUTPUT.open(s, ios::in);
				int numberOfLines = 0;
				float totalTimes = 0;
				string tp;
				while(getline(OUTPUT, tp))
				{
					totalTimes = totalTimes + std::stof(tp);
					numberOfLines++;

				}
				OUTPUT.close();
				averageIntervalTime = totalTimes/numberOfLines;
				

				OUTPUT.open(s, ios::app);
				OUTPUT << "The average time is: " << averageIntervalTime << endl;
				OUTPUT.close();

				intervalCounter++;
				totalTimeEndStart = std::chrono::system_clock::now();	
			}

			imshow("video player", grayMyImage);
			char c = (char)waitKey(25);
			if(c == 27)
			{
				break;
			}
			continue;	
		}
		


	}
	cap.release();
	return 0;
}
