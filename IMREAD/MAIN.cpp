#include <iostream>
#include <ostream>
#include <string>
#include "ctime"
#include "dos.h"
#include "stdio.h"
#include <stdlib.h>

#include <stddef.h>
#include <conio.h>
#include "math.h"
#include "vector"
#include <uEye.h>
#include <uEye_tools.h>
#include <ueye_deprecated.h>
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "stdafx.h"
#include "windows.h"
#include <array>
#include <opencv2/imgproc.hpp>
#include <algorithm>
//#include <unistd.h>
#include <uEyeCaptureInterface.h>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <map>
#include <iomanip>
#include <functional>
#include <opencv2/core/cuda.hpp>
#include "opencv2/objdetect/objdetect.hpp"
//for Gaussfitting




using namespace cv;  
using namespace std;


short stop=0;
char key;
/// Global variables für Canny
const int edgeThresh = 1;//ok
const int lowThreshold = 15 ;
const int const max_lowThreshold = 100;
const int ratio1 = 2;//war 3 ist ok oder vllt 2
const int kernel_size = 3;//gut so

const int img_width = 3840;
const int img_height = 2748;
const int img_bpp = 8;

Mat im_gray;
Mat im_bw;
Mat im_contour; 
Mat dst;
Mat im_bw_inv;
Mat im_rgb;
Mat image;



vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
RNG rng(12345);



string window_name_rgb = "Laser Beam in RGB";
string window_name_bw = "Laser Beam in BW";
string window_name_contour = "Laser Beam: Contours";
string window_name_dst = "rausfinden dst";
string window_name_bw_inv = "Invert BW";
string window_name_from_cam = "Image from Camera";
string im_name = "D:/Bild5";
string snap = "D:/Documents/Visual Studio 2012/Projects/IMREAD/Release/snap_BGR8";
string im_extension = ".jpg";
string bw = "_bw";


vector<Point2f> findCenterMassCenter()
{
	// Reduce noise with a kernel 3x3
	blur( im_gray, im_contour, Size(3,3) ); //берет  изображение im_gray и записывает его в Mat im_contour

	Canny( im_bw, im_contour, lowThreshold, lowThreshold*ratio1, kernel_size );

	//findContours( im_contour, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	findContours( im_bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	dst = Scalar::all(0);  // Using Canny's output as a mask, we display our resultt

	vector<Moments> mu(contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{ mu[i] = moments( contours[i], false ); }

	///  Get the mass centers:
	vector<Point2f> mc( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{ mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }


	Mat drawing = Mat::zeros( im_bw.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
		circle( drawing, mc[i], 4, color, -1, 8, 0 );
		/// Calculate the area with the moments 00 and compare with the result of the OpenCV function
		// printf("\t Info: Area and Contour Length \n");
	}

	for( int i = 0; i< contours.size(); i++ )
	{
		// printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
		circle( drawing, mc[i], 4, color, -1, 8, 0 );
	}
	cout<<"Schwerpunkt des Objektes"<<mc<<endl;
	//  imshow( window_name_contour, drawing );
	return mc;
}




Point findCenterPointContours()
{
	auto minX = NULL;
	auto minY = NULL;
	auto maxX = NULL;
	auto maxY = NULL;

	for (int i = 0; i < contours.size(); i++)
	{
		for (int j = 0; j < contours[i].size(); j++)
		{
			int x = contours[i][j].x;
			int y = contours[i][j].y;

			if (minX == NULL || x < minX)
			{
				minX = x;
			}
			if (maxX == NULL || x > maxX)
			{
				maxX = x;
			}
			if (minY == NULL || y < minY)
			{
				minY = y;
			}
			if (maxY == NULL || y > maxY)
			{
				maxY = y;
			}
		}
	}
	int centerX = (maxX + minX) / 2;
	int centerY = (maxY + minY) / 2;
	cout << centerX << ", " << centerY << " Center of Laser Beam" << endl;
	return Point(centerX, centerY);
}

//Point findCenterPointIntens(Mat)
//		{  
//	vector<int> VectorX(im_bw_inv.cols,0);
//	vector<int> VectorY(im_bw_inv.rows,0);	
//	int maxX = 0;
//	int maxY = 0;
//	
//	vector<int>::iterator itX;
//	vector<int>::iterator itY;
//
//	for(int i = 0; i < im_bw_inv.cols; i++) 
//	{	for(int j = 0; j < im_bw_inv.rows; j++)
//		{	int sumX += sumX;
//		
//			VectorX.push_back(sumX) += sumX;
//			
//			//unsinnint sumX = im_bw_inv.at<uint>(i,j);
//			
//		if (sumX <= maxX) { maxX = sumX; }
//			cout << "Centre X" << maxX <<endl;
//			//do{maxX = pixel
//			
//
//		//	maxY += im_bw_inv.at<uchar>(i,j);
//			//VectorY[j] += VectorY[j];
//			/*itY = VectorY.begin();
//			itX = VectorX.insert ( itX , im_bw_inv.at<uchar>(i,j) );*/
//
//		/*	itY = VectorY.begin();
//			itY = VectorY.insert(itY, im_bw_inv.at<uchar>(i, j));*/
//		}
//	
//	}
//
//    /*	 X = max_element(begin(VectorX), end(VectorX));
//		 Y = max_element(begin(VectorY), end(VectorY));
//		
//		X + = im_bw_inv.at<uchar>(i,j);}*/
////				
////for (itX = VectorX.begin(); itX != VectorX.end(); ++itX)
////			{
////				if (maxX < *itX)
////				        maxX = *itX;
////			}
////
////for (itY = VectorY.begin(); itY != VectorY.end(); ++itY)
////			{
////				if (maxY < *itY)
////				        maxY = *itY;
////			}
//
//
//	cout << maxX << ", " << maxY <<  " Center of Laser Beam from Intensivity of BW Image " << endl;
//	return Point(maxX, maxY);
//}

//void Histogrambuild( )
//{	Mat  src, dst;
//
//	
//	/// Load image
//	String imageName("D:/Bild4.jpg"); // by default
//	src = imread(imageName, IMREAD_COLOR);
//	
//
//	/// Separate the image in 3 places ( B, G and R )
//	vector<Mat> bgr_planes;
//	split(src, bgr_planes);
//
//	/// Establish the number of bins
//	int histSize = 256;
//
//	/// Set the ranges ( for B,G,R) )
//	float range[] = {0, 256};
//	const float* histRange = {range};
//
//	bool uniform = true;
//	bool accumulate = false;
//
//	Mat b_hist, g_hist, r_hist;
//
//	/// Compute the histograms:
//	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
//	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
//	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
//
//	// Draw the histograms for B, G and R
//	int hist_w = 1024;
//	int hist_h = 800;
//	int bin_w = cvRound((double) hist_w / histSize);
//
//	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
//
//	/// Normalize the result to [ 0, histImage.rows ]
//	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//
//	/// Draw for each channel EIGENTLICH UNNOETIG  SPATER LOESCHEN
//	for (int i = 1; i < histSize; i++)
//	{
//		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
//		     Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
//		     Scalar(255, 0, 0), 2, 8, 0);
//		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
//		     Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
//		     Scalar(0, 255, 0), 2, 8, 0);
//		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
//		     Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
//		     Scalar(0, 0, 255), 2, 8, 0);
//	}
//
//	/// Display
//	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
//	imshow("calcHist Demo", histImage);
//
//	waitKey(0);
//}

//void createGaussian(Size& image1, Mat & output, int uX, int uY, float sigmaX, float sigmaY, float amplitude = 1.0f)
//{
//	Mat temp = Mat(image1, CV_32F);
//	for (int row = 0; row < im_bw.rows; row++)
//	{
//		for (int col = 0; col < im_bw.cols; col++)
//		{
//			float x = (col-uX)*((float)col-uX)/(2.0f*sigmaX*sigmaX);
//			float y = (row-uY)*((float)row-uY)/(2.0f*sigmaY*sigmaY);
//			float value = amplitude*exp(-(x+y));
//			temp.at<float>(row,col) = value;
//		}
//
//	}
//	normalize(temp,temp,0.0f,1.0f,NORM_MINMAX);
//	output = temp;
//}

Point2d findCenterPointHist(const Mat image)

{
	double CenterX = 0;
	double CenterY = 0;
	double meanX, meanY;
	double varianceX, varinaceY;
	vector<double> VectorX(image.cols, 0);
	vector<double> VectorY;

	for (int row = 0; row < image.rows; row++)
	{
		double coloredCols = 0;
		for (int col = 0; col < image.cols; col++)
		{
			int pixelValue = image.at<uchar>(row, col);
			if (pixelValue == 1)
			{
				coloredCols += 1;
				VectorX.at(col) = VectorX.at(col) + 1;
			}
		}
		VectorY.push_back(coloredCols);
	}

	double sumVertical = accumulate(VectorX.begin(), VectorX.end(), 0);
	double sumHorizontal = accumulate(VectorY.begin(), VectorY.end(), 0);
	if (sumHorizontal != sumVertical)
	{
		cout << "Error counting pixels" << endl;
	}


	double MeanValuePixels = sumVertical / 2;
	double sumY = 0;
	double sumX = 0;
	int itY ;
	int itX ;


	for (itY = 0; itY < sumVertical; itY++)
	{
		sumY += VectorY.at(itY);
		CenterY = itY;
		if (sumY >= MeanValuePixels) break;
	}

	cout << CenterY << "Integrierte CenterY" << endl;


	for (itX = 0; itX < sumHorizontal; itX++)
	{
		sumX += VectorX.at(itX);
		CenterX = itX;
		if (sumX >= MeanValuePixels) break;
	}

	cout << CenterX << "Integrierte CenterX" << endl;

	//meanX = (accumulate(VectorX.begin(), VectorX.end(), 0)) / image.cols;
	//for (int row = 0; row < image.rows; row++)
	//{
	//	VectorX[row] = VectorX[row] - meanX;
	//	VectorX[row] *= (VectorX[row]);
	//}
	//varianceX = (accumulate(VectorX.begin(), VectorX.end(), 0)) ^ 2 / image.cols;
	////cout << "variance " << varianceX << endl;


	return Point2d(sumVertical, sumHorizontal);
}

map<int, int> computeHistogram(const Mat& image)
{
	map<int, int> histogram;

	for ( int row = 0; row < image.rows; ++row)
	{
		for ( int col = 0; col < image.cols; ++col)
		{
			++histogram[(int)image.at<uchar>(row, col)];

		}
	}

	return histogram;
}

void printHistogram(const map<int, int>& histogram) //create vector  with 0 and 1 for x and y derection
{
	map<int, int>::const_iterator histogram_iter;
	cout << "\n------------------\n";
	for (histogram_iter = histogram.begin(); histogram_iter != histogram.end(); ++histogram_iter)
	{
		cout << setw(5) << histogram_iter->first << " : " << histogram_iter->second << "\n";
	}
	cout << "------------------\n";
}

//void GlobalTrend::gd_slope(float &slope_mu, float &slope_sigma) {
//int nmax = residual.size();
//float tmp_sigma = 0.f, tmp_mu = 0.f;
//for(int i = 0; i < nmax; ++i) {
//tmp_sigma += 2.f*residual[i]*exp(-pow(x_mu_sigma[i],2))/sqrt(M_PI)*(-x_mu_sigma[i]/sigma);
//tmp_mu += 2.f*residual[i]*exp(-pow(x_mu_sigma[i],2))/sqrt(M_PI)*(-1.f/sqrt(2.f*pow(sigma,2)));
//}
//slope_sigma = tmp_sigma;
//slope_mu = tmp_mu;
//return;
//}


int main()
{

	HIDS hCam = 1;
#define CAPTURE_WIDTH  768
#define CAPTURE_HEIGHT 576

	//Allocazione della matrice immagine per il frame catturato dalla telecamera
	Mat im_snap(CAPTURE_HEIGHT, CAPTURE_WIDTH,CV_8UC3);

	//puntatori memoria
	char* m_pcImageMemory;
	int m_lMemoryId;
	//char* imgMem;
	//int memId;

	//Apre Camera con ID 1
	int BITS_PER_PIXEL = 16;
	int pWidth = CAPTURE_WIDTH;
	int pHeight = CAPTURE_HEIGHT; 
	SENSORINFO sensor_info;
	CAMINFO camera_info;

	double FPS,NEWFPS;
	FPS = 3; // 2.13 ist max
	is_SetFrameRate(hCam,FPS,&NEWFPS);

	double Exposure = 300; // Belichtungszeit (in ms)
	is_Exposure(hCam, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*) &Exposure, sizeof(Exposure));

	is_SetGamma(hCam,100); // Default = 100, corresponds to a gamma value of 1.0
	is_Focus (hCam, FOC_CMD_SET_DISABLE_AUTOFOCUS, NULL, 0);
	is_SetHardwareGain (hCam, 1, 0, 0, 0);

	is_SetDisplayMode (hCam, IS_SET_DM_DIB); //Bitmap-Modus
	is_SetColorMode (hCam, IS_CM_RGB8_PACKED);
	//is_SetColorMode (hCam, IS_CM_MONO8);              GROESSTER FRAGE
	is_SetImageSize (hCam, pWidth, pHeight);

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


	//im_rgb = imread(im_name + im_extension, CV_LOAD_IMAGE_COLOR);
	//namedWindow(window_name_rgb, CV_WINDOW_AUTOSIZE);
	//imshow(window_name_rgb, im_rgb);

	//cvtColor(im_rgb, im_gray, CV_RGB2GRAY); //RGB to grayscale
	//im_bw = im_gray > 128; //convert to binary

	//imwrite((im_name + bw + im_extension), im_bw); //save to disk
	//namedWindow(window_name_bw, CV_WINDOW_AUTOSIZE);
	//imshow(window_name_bw, im_bw);
	//cout << "SIZE" << im_bw.size << endl;

	//printHistogram(computeHistogram(im_rgb));

	//TRESHOLD BINARY INV
	double threshold_value = 1;
	double max_BINARY_value = 1;
	//threshold ( im_bw, im_bw_inv, threshold_value, max_BINARY_value,THRESH_BINARY_INV);// ein SWsBild, wo  Laserstrahl schwarz ist
	//namedWindow ( window_name_bw_inv, CV_WINDOW_AUTOSIZE );
	//imshow ( window_name_bw_inv, im_bw_inv);
	//findCenterMassCenter();//Methode mit Objekt Schwerpunkt
	//findCenterPointContours();//Methode mit Contouren
	//findCenterPointHist(im_bw); //Methode mit Integralen und Histogramen



	if(is_InitCamera (&hCam, NULL)!= IS_SUCCESS)
	{
		cout << " Initialisierungsproblem: Kamera ausstecken-einstecken?" << endl;
		//		return 0;
	}



	//Pulizia memoria da foto precedenti
	if (hCam != 0){
		is_FreeImageMem (hCam,m_pcImageMemory,m_lMemoryId);
		is_ExitCamera(hCam);
	}

	//inizializzazione della telecamera 
	int initcamera = is_InitCamera(&hCam, NULL);
	if(initcamera != IS_SUCCESS)
	{
		cout << endl << "Initialisierung der Camera ist möglich!" << endl;
		exit(-1);
	}

	// Acquisisce informazioni riguardanti la telecamera
	int camerainfo = is_GetCameraInfo (hCam, &camera_info);
	if(camerainfo != IS_SUCCESS)
	{
		printf("Impossibile acquisire le informazioni della telecamera");
		exit(-1);
	} 
	// Acquisisce informazioni riguardanti il sensore della telecamera
	int sensorinfo = is_GetSensorInfo (hCam, &sensor_info);
	if(sensorinfo != IS_SUCCESS)
	{
		printf("Impossibile acquisire le informazioni del sensore");
		exit(-1);
	}

	//Output informazioni camera/sensore
	cout<<endl<<"<<< CARATTERISTICHE DELLA TELECAMERA COLLEGATA >>>"<<endl;
	cout<<"Numero seriale: " << camera_info.SerNo << endl;
	cout << "Produttore: " << camera_info.ID << endl;
	cout << "Modello: " << sensor_info.strSensorName << endl;
	cout << "Dimensioni massime per l'immagine: " << sensor_info.nMaxWidth << "x" << sensor_info.nMaxHeight << endl << endl;


	//Imposta la modalità di colore BGR24 
	int colormode = is_SetColorMode(hCam, IS_CM_BGR8_PACKED);
	//int colormode = is_SetColorMode(hCam, IS_SET_CM_RGB24);
	if(colormode != IS_SUCCESS)
	{
		printf("Impossibile impostare il modo di colore");
		exit(-1);
	}

	//imposta dimensioni immagini che voglio catturare
	int pXPos = (sensor_info.nMaxWidth);
	int pYPos = (sensor_info.nMaxHeight);

	//Inizializzazione Memoria camera
	int rit = is_AllocImageMem (hCam,pXPos,pYPos, 24, &m_pcImageMemory, &m_lMemoryId);
	if(rit != IS_SUCCESS)
	{
		cout<<endl<<"IMPOSSIBILE INIZIALIZZARE LA MEMORIA"<<endl;
		system("PAUSE");
		exit(-1);
	}
	cout<<endl<<"Memoria inizializzata"<<endl;

	//attivazione della locazione di memoria
	int rat = is_SetImageMem (hCam, m_pcImageMemory, m_lMemoryId);
	if(rat != IS_SUCCESS)
	{
		cout<<endl<<"IMPOSSIBILE ATTIVARE LA MEMORIA"<<endl;
		system("PAUSE");
		exit(-1);
	}
	cout<<endl<<"Memoria Attivata"<<endl;

	//impostazioni correzioni di colore
	double strenght_factor = 1.0;
	int colorcorrection = is_SetColorCorrection(hCam, IS_CCOR_ENABLE, &strenght_factor);

	//impostazioni correzione del bianco
	double pval = 1;
	int whiteb = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &pval, 0);

	//impostazione della correzione guadagno
	double gval = 1;
	int gains = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_GAIN, &gval, 0);


	//inizio fase cattura immagine
	int dummy;
	char *pMem, *pLast;

	//ciclo di ripetizione
	//for (int i=0;i<10;i++)
	//{
	short stop = 0;

	while (stop==0)
	{
		int freeze = is_FreezeVideo(hCam, IS_WAIT);
		if(freeze != IS_SUCCESS)
		{
			cout<<endl<<"IMPOSSIBILE ACQUISIRE DALLA TELECAMERA"<<endl;
			system("PAUSE");
			exit(-1);
		}



		if (freeze == IS_SUCCESS)
		{
			int m_Ret = is_GetActiveImageMem(hCam, &pLast, &dummy);
			int n_Ret = is_GetImageMem(hCam, (void**)&pLast);
		}

		IplImage* tmpImg = cvCreateImageHeader(cvSize (pXPos, pYPos), IPL_DEPTH_8U,3); 
		tmpImg->imageData = m_pcImageMemory;
		im_snap = cvarrToMat(tmpImg);
		imshow("SNAPSHOT",im_snap);
		//stop = GetAsyncKeyState(VK_LSHIFT);
		waitKey(0);
		stop = GetAsyncKeyState(VK_LSHIFT);
	}
	//chiusura e pulizia della telecamera
	int en = is_ExitCamera(hCam);
	if (en == IS_SUCCESS)
	{
		cout << endl << "Camera chiusa correttamente" << endl;
	}

	return 0;
}





//////////////
//////////////	short stop=0;
//////////////
//////////////	while (stop==0)
//////////////	{		
//////////////		if(is_FreezeVideo(hCam, IS_WAIT) == IS_SUCCESS){
//////////////			void *pMemVoid; //pointer to where the image is stored
//////////////			is_GetImageMem (hCam, &pMemVoid);
//////////////			IplImage * img;
//////////////			img=cvCreateImage(cvSize(img_width, img_height), IPL_DEPTH_8U, 1); 
//////////////			img->nSize=sizeof(IplImage);
//////////////			img->ID=0;
//////////////			img->nChannels=1;
//////////////			img->alphaChannel=0;
//////////////			img->depth=8;
//////////////			img->dataOrder=0;
//////////////			img->origin=0;
//////////////			img->align=4;	// egal
//////////////			img->width=img_width;
//////////////			img->height=img_height;
//////////////			img->roi=NULL;
//////////////			img->maskROI=NULL;
//////////////			img->imageId=NULL;
//////////////			img->tileInfo=NULL;
//////////////			img->imageSize=img_width*img_height;
//////////////			img->imageData=(char*)pMemVoid;  //the pointer to imagaData
//////////////			img->widthStep=img_width;
//////////////			img->imageDataOrigin=(char*)pMemVoid; //and again



//////////////	//findCenterMassCenter();//Methode mit Objekt Schwerpunkt
//////////////	//findCenterPointContours();//Methode mit Contouren
//////////////	//findCenterPointHist(im_bw); //Methode mit Integralen und Histogramen
//////////////
//////////////			// Resize img (für cvShowImage)
//////////////			IplImage* img_resized = cvCreateImageHeader(cvSize);
//////////////			double resize_factor = 1.3;//0.8
//////////////			img_resized = cvCreateImage(cvSize((int)(resize_factor*img->width), (int)(resize_factor*img->height)), img->depth, img->nChannels);
//////////////			cvResize(img, img_resized);
//////////////			//im_rgb =  img_resized;
//////////////

//////////////
//////////////			resizeWindow("A", img_width/3, img_height/4);

//////////////	else
//////////////	{
//////////////		cout << "ERROR FREEZE" << endl;
//////////////	}
//////////////	//this_thread::sleep_for(zeit);
//////////////	stop = GetAsyncKeyState(VK_LSHIFT);
//////////////}