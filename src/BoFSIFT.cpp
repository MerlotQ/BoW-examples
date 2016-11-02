// BoFSIFT.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

int main( int argc, char** argv )
{	
#if DICTIONARY_BUILD == 1

	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filename = new char[100];		
	//to store the current input image
	Mat input;	

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;
	//To store the SIFT descriptor of current image
	Mat descriptor;
	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;
	//The SIFT feature extractor and descriptor
	cv::SiftDescriptorExtractor detector;

	//I select 20 (1000/50) images from 1000 images to extract feature descriptors and build the vocabulary
	for(int f=0;f<999;f+=50){		
		//create the file name of an image
		sprintf(filename,"G:\\testimages\\image\\%i.jpg",f);
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		//detect feature points
		detector.detect(input, keypoints);
		//compute the descriptors for each keypoint
		detector.compute(input, keypoints,descriptor);		
		//put the all feature descriptors in a single Mat object 
		featuresUnclustered.push_back(descriptor);		
		//print the percentage
		printf("%i percent done\n",f/10);
	}	


	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=200;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	Mat dictionary=bowTrainer.cluster(featuresUnclustered);	
	//store the vocabulary
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	
#else
	//Step 2 - Obtain the BoF descriptor for given image/video frame. 

    //prepare BOW descriptor extractor from the dictionary    
	Mat dictionary; 
	FileStorage fs("/home/bobo/code/BoW_examples/data/dictionary.yml", FileStorage::READ);
        if(!fs.isOpened())
        {
           cerr << "Failed to open settings file  "<< endl;
           exit(-1);
        }
	fs["vocabulary"] >> dictionary;
	fs.release();	

	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	//To store the image file name
	char * filename = new char[100];
	//To store the image tag name - only for save the descriptor in a file
	char * imageTag = new char[10];

	//open the file to write the resultant descriptor
	FileStorage fs1("/home/bobo/code/BoW_examples/data/descriptor.yml", FileStorage::WRITE);	
	double dataset_num =4;
	for(int i = 1; i < dataset_num+1;  i++)
		{
	//the image file with the location. change it according to your image file location
	sprintf(filename,"/home/bobo/code/BoW_examples/data/%01d.jpg",i);
	//read the image
	Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)cvGetTickCount();
	
	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;		
	//Detect SIFT keypoints (or feature points)
	detector->detect(img,keypoints);
	//To store the BoW (or BoF) representation of the image
	Mat bowDescriptor;		
	//extract BoW (or BoF) descriptor from given image
	bowDE.compute(img,keypoints,bowDescriptor);

  double bow_des_cost =  ((double)cvGetTickCount() - t)/((double)cvGetTickFrequency()*1000.);
  cout<<"extract BoW descriptor cost "<<bow_des_cost<<"ms"<<endl;

	//prepare the yml (some what similar to xml) file
	sprintf(imageTag,"img%01d",i);
	//write the new BoF descriptor to the file
	fs1 << imageTag << bowDescriptor;		//直方图归一化

	//You may use this descriptor for classifying the image.
	}
	//release the file storage
	fs1.release();

#endif
	printf("done\n");	
    return 0;
}
