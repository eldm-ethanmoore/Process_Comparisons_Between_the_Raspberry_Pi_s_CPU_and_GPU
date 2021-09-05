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

Mat copyValues(Mat image, float grayArray[]);
float* gpuOperation(bool &isAdd, float intArray[], float floatArray[], int size);

int main( int argc, char** argv)
{

	auto start = high_resolution_clock::now();

	Mat image;

	image = imread( argv[1], CV_LOAD_IMAGE_COLOR);

	if(argc !=2 || !image.data)
	{
		printf( "No image data \n" );
		return -1;
	}

	int totalPixels = image.total();

	float *B;
	B = (float*) malloc(totalPixels*sizeof(float));
	float *G;
	G = (float*) malloc(totalPixels*sizeof(float));
	float *R;
	R = (float*) malloc(totalPixels*sizeof(float));

	float *BF;
	BF = (float*) malloc(totalPixels*sizeof(float));
	std::fill_n (BF, totalPixels, .07);
	float *GF;
	GF = (float*) malloc(totalPixels*sizeof(float));
	std::fill_n (GF, totalPixels, .72);
	float *RF;
	RF = (float*) malloc(totalPixels*sizeof(float));
	std::fill_n (RF, totalPixels, .21);

	int counter = 0;

	for(int row=0;row<image.rows;row++)
	{
		for(int col=0;col<image.cols;col++)
		{
			Vec3b pixel = image.at<Vec3b>(row,col);
			B[counter]=pixel.val[0];
			G[counter]=pixel.val[1];
			R[counter]=pixel.val[2];
			counter++;
		}
	}
	
	
	bool adding = false;

	float *BMul;
	BMul = (float*) malloc(totalPixels*sizeof(float));
	float *GMul;
	GMul = (float*) malloc(totalPixels*sizeof(float));
	float *RMul;
	RMul = (float*) malloc(totalPixels*sizeof(float));

	BMul=gpuOperation(adding, B, BF, totalPixels);
	free(B);
	free(BF);
	GMul=gpuOperation(adding, G, GF, totalPixels);
	free(G);
	free(GF);
	RMul=gpuOperation(adding, R, RF, totalPixels);
	free(R);
	free(RF);
	
	adding = true;

	float *BGTotal;
	BGTotal = (float*) malloc(totalPixels*sizeof(float));
	float *total;
	total= (float*) malloc(totalPixels*sizeof(float));


	BGTotal=gpuOperation(adding, BMul, GMul, totalPixels);
	free(BMul);
	free(GMul);
	total=gpuOperation(adding, BGTotal, RMul, totalPixels);	
	free(RMul);
	free(BGTotal);	

	image=copyValues(image, total);
		
	free(total);
	
	Mat splitChannels[3];
	split(image, splitChannels);

	imshow( "Display Image", splitChannels[0]);
	//imwrite("GrayScaleImage.jpg", splitChannels[0]);
	waitKey(0);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

	return 0;
}

float* gpuOperation(bool &isAdd, float intArray[], float floatArray[], int size){

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


    cl::Context context({default_device});

    cl::Program::Sources sources;

    std::string kernel_code;
    if(isAdd){ 
    kernel_code=
            "   void kernel math(global const float* A, global const float* B, global float* C){       "
            "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
            "   }                                                                               ";
    }
    if(!isAdd){
    kernel_code=
            "   void kernel math(global const float* A, global const float* B, global float* C){       "
            "       C[get_global_id(0)]=A[get_global_id(0)]*B[get_global_id(0)];                 "
            "   }                                                                               ";
    }
    sources.push_back({kernel_code.c_str(),kernel_code.length()});


    cl::Program program(context,sources);
    
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
    

    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(float)*size);

    cl::CommandQueue queue(context,default_device);

    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float)*size,intArray);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float)*size,floatArray);

    cl::Kernel kernel_add=cl::Kernel(program,"math");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
    
    float *C;
    C = (float*) malloc(size*sizeof(float));

    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(float)*size,C);

    return C;
}

Mat copyValues(Mat image, float grayArray[]){
        int counter = 0;	
	for(int row=0;row<image.rows;row++)
		{
			for(int col=0;col<image.cols;col++)
			{
				image.at<cv::Vec3b>(row, col)[0] = (int) grayArray[counter];
				counter++;
			}
		}

	return image;

}
