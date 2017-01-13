#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <ctime>
#include <uEye.h>
#include <uEye_tools.h>
#include <ueye_deprecated.h>
#include <wchar.h>
#include <locale.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "stdafx.h"
#include <windows.h>


using namespace cv;  
using namespace std;
char key;
/// Global variables für Canny
int edgeThresh = 1;//ok
int lowThreshold = 15 ;
int const max_lowThreshold = 100;
int ratio = 2;//war 3 ist ok oder vllt 2
int kernel_size = 3;//gut so
char* window_name_rgb = "Laser Beam in RGB";
char* window_name_bw = "Laser Beam in BW";
char* window_name_contour = "Laser Beam: Contours";
char* window_name_dst = "rausfinden dst";
char* window_name_bw_inv = "Invert BW";
char* window_name_from_cam = "Image from Camera";

Mat im_gray;
Mat im_bw;
Mat im_contour; // hier wird image mit countouren geladen
Mat dst;
Mat im_bw_inv;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
RNG rng(12345);



void CannyThreshold(int, void*)
{
  // Reduce noise with a kernel 3x3
 // blur( im_gray, im_contour, Size(3,3) ); //берет  изображение im_gray и записывает его в Mat im_contour
 
	Canny( im_bw, im_contour, lowThreshold, lowThreshold*ratio, kernel_size );
 
  // Find contours
  //findContours( im_contour, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  findContours( im_bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  /// Using Canny's output as a mask, we display our resultt
  dst = Scalar::all(0);

//  for(int i= 0; i < contours.size(); i++)
//{
//    for(int j= 0; j < contours[i].size();j++) // run until j < contours[i].size();
//    {
//  cout << contours[i][j] << " Contours" << endl;
// 	}
//	  }

Mat drawing = Mat::zeros( im_contour.size(), CV_8UC3 ); //Matrix mit Contouren
 for( int i = 0; i < contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() ); //war Point(5,5)
   }
  
  imshow( window_name_contour, drawing );

  cout << contours.size()<< " Contours SIZE" << endl;
 }


//////Milenas Prog
////void CameraInit()
////{
////	//string file_path = "D:/Documents/Visual Studio 2012/Projects/CentreDetection/CentreDetection.txt";
////	//ofstream myfile;
////	//myfile.open (file_path, ios::trunc); // Inhalt der vorhandenen Datei löschen
////	//myfile	<< "Datum und Uhrzeit\t" << "Leistung" << endl << endl;
////	//myfile.close();
////
//////	time_t t;
////	//struct tm now;
////
////	HIDS hCam = 0;
////	char* imgMem;
////	int memId;
////	HWND hwnd;
////	int nMode;
////
////	if(is_InitCamera (&hCam, NULL)!= IS_SUCCESS)
////	{
////		cout << " Initialisierungsproblem: Kamera ausstecken-einstecken?" << endl;
//////		return 0;
////	}
////	//const int s.width=3840, s.height=2748;
////	const int img_bpp=8;
////	is_AllocImageMem (hCam, s.width, s.height, img_bpp, &imgMem, &memId); // Um den DIB-Modus zu verwenden muss ein Speicher angelegt werden
////	is_SetImageMem (hCam, imgMem, memId);//Speicher aktiv setzen
////	is_SetDisplayMode (hCam, IS_SET_DM_DIB); //Bitmap-Modus
////	//is_RenderBitmap( hCam, nMemID, hwnd, IS_RENDER_FIT_TO_WINDOW); // ein Bild wird aus einem Bildspeicher in dem angegebenen Fenster ausgegeben
////	is_SetColorMode (hCam, IS_CM_MONO8);
////	is_SetImageSize (hCam, s.width, s.height);
////
////	double disable = 0;
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_GAIN, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_FRAMERATE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SHUTTER, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_GAIN, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_WHITEBALANCE,&disable,0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_SHUTTER, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER,&disable,0);
////	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_FRAMERATE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_REFERENCE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_ANTI_FLICKER_MODE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_BACKLIGHT_COMP, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_CONTRAST_CORRECTION, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_SHUTTER_PHOTOM, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_SKIPFRAMES, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_SKIPFRAMES, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_GAIN_PHOTOM, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_HYSTERESIS, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_REFERENCE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_GAIN_MAX, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_SHUTTER_MAX, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_SPEED, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_OFFSET, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_GAIN_RANGE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_SPEED, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_ONCE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_BRIGHTNESS_ONCE, &disable, 0);
////	is_SetAutoParameter (hCam, IS_SET_AUTO_HYSTERESIS, &disable, 0);
////
////	double FPS,NEWFPS;
////	FPS = 3; // 2.13 ist max
////	is_SetFrameRate(hCam,FPS,&NEWFPS);
////
////	double Exposure = 300; // Belichtungszeit (in ms)
////	is_Exposure(hCam, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*) &Exposure, sizeof(Exposure));
////
////	is_SetGamma(hCam,100); // Default = 100, corresponds to a gamma value of 1.0
////	is_Focus (hCam, FOC_CMD_SET_DISABLE_AUTOFOCUS, NULL, 0);
////	is_SetHardwareGain (hCam, 1, 0, 0, 0);
////
////	//int ii = 0;
////	//short stop=0;
////	
////}
////	
//			
//			
//
//
Point findCenterPoint()
{  
	int minX = NULL;
	int minY = NULL;
	int maxX = NULL;
	int maxY = NULL;

	for(int i = 0; i < contours.size(); i++) {
		for(int j = 0; j < contours[i].size(); j++) {
			int x = contours[i][j].x;
			int y = contours[i][j].y;

			if(minX == NULL || x < minX) {
				minX = x;
			}
			if(maxX == NULL || x > maxX) {
				maxX = x;
			}
			if(minY == NULL || y < minY) {
				minY = y;
			}
			if(maxY == NULL || y > maxY) {
				maxY = y;
			}
		}
	}
	int centerX = (maxX + minX) / 2;
	int centerY = (maxY + minY) / 2;
	cout << centerX << ", " << centerY <<  " Center of Laser Beam" << endl;
	return Point(centerX, centerY);
}


void CameraInit()

{	HIDS hCam = 0;
	char* imgMem;
	int memId;

	
	const int img_width=3840, img_height=2748, img_bpp=8;
	is_AllocImageMem (hCam, img_width, img_height, img_bpp, &imgMem, &memId);
	is_SetImageMem (hCam, imgMem, memId);
	is_SetDisplayMode (hCam, IS_SET_DM_DIB); //Bitmap-Modus
	is_SetColorMode (hCam, IS_CM_MONO8);
	is_SetImageSize (hCam, img_width, img_height);

	double disable = 0;
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_GAIN, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_FRAMERATE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SHUTTER, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_GAIN, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_WHITEBALANCE,&disable,0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_SHUTTER, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER,&disable,0);
	is_SetAutoParameter (hCam, IS_SET_ENABLE_AUTO_SENSOR_FRAMERATE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_REFERENCE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_ANTI_FLICKER_MODE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_BACKLIGHT_COMP, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_CONTRAST_CORRECTION, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_SHUTTER_PHOTOM, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_SKIPFRAMES, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_SKIPFRAMES, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_SENS_AUTO_GAIN_PHOTOM, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_HYSTERESIS, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_REFERENCE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_GAIN_MAX, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_SHUTTER_MAX, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_SPEED, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_OFFSET, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_GAIN_RANGE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_SPEED, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_WB_ONCE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_BRIGHTNESS_ONCE, &disable, 0);
	is_SetAutoParameter (hCam, IS_SET_AUTO_HYSTERESIS, &disable, 0);

	double FPS,NEWFPS;
	FPS = 3; // 2.13 ist max
	is_SetFrameRate(hCam,FPS,&NEWFPS);

	double Exposure = 300; // Belichtungszeit (in ms)
	is_Exposure(hCam, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*) &Exposure, sizeof(Exposure));

	is_SetGamma(hCam,100); // Default = 100, corresponds to a gamma value of 1.0
	is_Focus (hCam, FOC_CMD_SET_DISABLE_AUTOFOCUS, NULL, 0);
	is_SetHardwareGain (hCam, 1, 0, 0, 0);

	int ii = 0;
	short stop=0;

	while (stop==0)
	{		
		if(is_FreezeVideo(hCam, IS_WAIT) == IS_SUCCESS){
			void *pMemVoid;				//pointer to where the image is stored
			is_GetImageMem (hCam, &pMemVoid);
			IplImage * img;
			img=cvCreateImage(cvSize(img_width, img_height), IPL_DEPTH_8U, 1); 
			img->nSize=sizeof(IplImage);
			img->ID=0;
			img->nChannels=1;
			img->alphaChannel=0;
			img->depth=8;
			img->dataOrder=0;
			img->origin=0;
			img->align=4;	// egal
			img->width=img_width;
			img->height=img_height;
			img->roi=NULL;
			img->maskROI=NULL;
			img->imageId=NULL;
			img->tileInfo=NULL;
			img->imageSize=img_width*img_height;
			img->imageData=(char*)pMemVoid;  //the pointer to imagaData
			img->widthStep=img_width;
			img->imageDataOrigin=(char*)pMemVoid; //and again

			//now you can use your img just like a normal OpenCV image	

			// Resize img (für cvShowImage)
			IplImage *img_resized;
			double resize_factor = 0.1;
			img_resized = cvCreateImage(cvSize((int)(resize_factor*img->width), (int)(resize_factor*img->height)), img->depth, img->nChannels);
			cvResize(img, img_resized);

			cvNamedWindow( "A", 1 );
			cvShowImage("A",img_resized);
			cv::waitKey(1);
		}
		stop = GetAsyncKeyState(VK_LSHIFT);
	}

	is_ExitCamera(hCam);
}


int main()
{	
	
string im_name="Bild1";
string im_extension = ".jpg";
string bw = "_bw";
cout << im_name + im_extension <<  endl;
Mat im_rgb;
im_rgb = imread (im_name + im_extension, CV_LOAD_IMAGE_COLOR);
	cout << im_rgb.rows << ";st " << im_rgb.cols << endl;
	namedWindow(window_name_rgb, CV_WINDOW_AUTOSIZE );
	
	imshow( window_name_rgb, im_rgb );
	
	cvtColor(im_rgb,im_gray,CV_RGB2GRAY); //RGB to grayscale

	im_bw = im_gray > 128; //convert to binary
	imwrite((im_name+bw+im_extension), im_bw); //save to disk
	
	namedWindow( window_name_bw, CV_WINDOW_AUTOSIZE );
	imshow( window_name_bw, im_bw );
	
	double threshold_value = 1;
	double max_BINARY_value = 255;

	threshold( im_bw, im_bw_inv, threshold_value, max_BINARY_value,THRESH_BINARY_INV);// ein SWsBild, wo  Laserstrahl schwarz ist
	namedWindow( window_name_bw, CV_WINDOW_AUTOSIZE );
	imshow( window_name_bw_inv, im_bw_inv );

	
	/// Create a matrix of the same type and size as im_gray (for dst)
	//dst.create( im_gray.size(), im_gray.type() );
	//short stop = 0; //for closing	MAIN
int rows = im_gray.rows;
int cols = im_gray.cols;
Size s = im_gray.size();
rows = s.height;
cols = s.width;
cout << s << "Size of Image" << endl;
cout << rows << " Rows" << endl;
cout << cols << " Cols" << endl;

CannyThreshold(0, 0);
findCenterPoint();


   waitKey(0);	
	//cvDestroyAllWindows();
    // Wait for a keystroke in the window
	//stop = GetAsyncKeyState(VK_LSHIFT); 
    return 0;
}

