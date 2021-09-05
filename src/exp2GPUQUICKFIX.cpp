#include <sstream>
#include <string>
#include <iostream>
#include <CL/cl.hpp>
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

float averageIntervalTime = 0;
int *frameOne;
int *frameTwo;
int *pixelChange;
int intervalCounter = 0;
int* gpuOperation(int frameOne[], int frameTwo[], int size);

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

			pixelChange = (int*) malloc(grayMyImage.total()*sizeof(int));

                        auto startComputationInterval = high_resolution_clock::now();
			pixelChange=gpuOperation(frameOne, frameTwo, grayMyImage.total());
			auto stopComputationInterval = high_resolution_clock::now();
			
			auto duration = duration_cast<microseconds>(stopComputationInterval - startComputationInterval);

			free(frameOne);
			free(frameTwo);
			counter = 0;
			for(int row=0;row<grayMyImage.rows;row++)
			{
				for(int col=0;col<grayMyImage.cols;col++)
				{
					
					if(pixelChange[counter]>=70)
					{
						grayMyImage.at<Vec3b>(row, col)[0] = 255;
					}
					counter++;
					
				}
			}
			free(pixelChange);
			stepNumber = 0;

			imshow("video player", grayMyImage);
			char c = (char)waitKey(25);


			fstream OUTPUT;
			std::stringstream ss;
			ss << "GPUInterval" << intervalCounter << ".txt";
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


int* gpuOperation(int frameOne[], int frameTwo[], int size){

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    
   /* 
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
   */ 
    cl::Platform default_platform=all_platforms[0];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    
   /* 
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
   */ 
    cl::Device default_device=all_devices[0];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


    cl::Context context({default_device});

    cl::Program::Sources sources;
 
    string kernel_code=
            "   void kernel math(global const int* A, global const int* C, global int* B){                     "
            "       B[get_global_id(0)]=abs(A[get_global_id(0)]-C[get_global_id(0)]);                               "
            "   }                                                                               ";

    sources.push_back({kernel_code.c_str(),kernel_code.length()});


    cl::Program program(context,sources);
    
    program.build({default_device});
    /*
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
    */

    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*size);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*size);

    cl::CommandQueue queue(context,default_device);

    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*size,frameOne);
    queue.enqueueWriteBuffer(buffer_C,CL_TRUE,0,sizeof(int)*size,frameTwo);

    cl::Kernel kernel_add=cl::Kernel(program,"math");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_C);
    kernel_add.setArg(2,buffer_B);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
   
    int *frameThreeDiff; 
    frameThreeDiff = (int*) malloc(size*sizeof(int));	

    queue.enqueueReadBuffer(buffer_B,CL_TRUE,0,sizeof(int)*size,frameThreeDiff);
/*	
    for(int i=0;i<1000;i++)
         {
		
         }
*/	
    return frameThreeDiff;
}



