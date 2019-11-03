/*
Copyright by:
  Sudiro
    [at] SudiroEEN@gmail.com
*/

#ifndef GMM_EM_HPP
#define GMM_EM_HPP

#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


class GMM{
private:
	int nKluster;
	int nData;
	int _dimensi;

	vector<Mat> _data_training;
	vector<Mat> mu_k;
	vector<Mat> Sigma_k;
	vector<Mat> _w_i_k;
	vector<Mat> _pdf_k;

	vector<double> _alpha_k;

private:
	vector<Mat> initialR;

private:
	string saveToYAML;
	string lutYAML;
public:
	Mat canvasYAML;
	
public:
    GMM();
	GMM(int nKluster, vector<Mat> dataset, vector<Mat> initialR_, string saveToYAML_, string lutYAML_, int row_, int col_);

	void initializeParam();
	void initializeLUT(int row, int col);

	void save2LUTyaml();
	
	bool isConvergence(vector<double> bobot_alpha_k, vector<Mat> _miyu, vector<Mat> _sigma, double& before_log);
	void train(int iterasi, bool saveit);
	double _PDF(Mat datum, Mat rerata, Mat covariance);
	
   	void loadConfig(string configYAML);
	int predict(const Mat& raw_pixel);
	
	
	
	enum STATE{
   	STATE_TRAIN = 0,
   	STATE_SEGMENT = 1,
   	STATE_OTHER = 2
   };

};

#endif
