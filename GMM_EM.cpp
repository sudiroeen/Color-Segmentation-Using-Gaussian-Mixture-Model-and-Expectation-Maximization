#include "GMM_EM.hpp"

GMM::GMM(int jmlKluster, vector<Mat> dataset, vector<Mat> initialR_, string saveToYAML_, string lutYAML_, int row_, int col_)
	:nKluster(jmlKluster), nData(dataset.size()), initialR(initialR_), lutYAML(lutYAML_),
	_data_training(dataset), saveToYAML(saveToYAML_)
{	
	initializeLUT(row_, col_);
	initializeParam();
	cout << "initial OK" << endl;
}

GMM::GMM(){
	
}

void GMM::initializeLUT(int row, int col){
	canvasYAML = Mat::zeros(row, col, CV_32SC1);
	FileStorage fs(lutYAML, FileStorage::WRITE);
	fs << "row" << row;
	fs << "col" << col;
	fs << "canvasYAML" << canvasYAML;
	fs.release();
}


void GMM::initializeParam(){
	_dimensi = initialR[0].rows;
	mu_k.resize(nKluster);
	Sigma_k.resize(nKluster);
	_alpha_k.resize(nKluster);

	_w_i_k.resize(nKluster);
	_pdf_k.resize(nKluster);

	for(int k=0; k<nKluster; k++){
		_w_i_k[k] = Mat::zeros(1, nData, CV_64FC1);
		_pdf_k[k] = Mat::zeros(1, nData, CV_64FC1);

		_alpha_k[k] = 1.0/nKluster;
	}

	mu_k = initialR;

	for(int k=0; k<nKluster; k++){
		Sigma_k[k] = Mat::zeros(initialR[0].rows, initialR[0].rows, CV_64FC1);
		for(int n=0; n<_data_training.size(); n++){
			Sigma_k[k] += (_data_training[n] - mu_k[k]) * (_data_training[n] - mu_k[k]).t();
		}
		Sigma_k[k] /= _data_training.size();
	}

	double sigma_alphaXpdf = 0.0;
	for(int k=0; k<nKluster; k++){
		double* _pix_wi_k = _w_i_k[k].ptr<double>(0);
		for(int n=0; n<nData; n++){
			_pix_wi_k[n] = _alpha_k[k] * _PDF(_data_training[n], mu_k[k], Sigma_k[k]);
			sigma_alphaXpdf += _pix_wi_k[n];
		}
	}

	for(int k=0; k<nKluster; k++){
		_w_i_k[k] /= sigma_alphaXpdf;
	}
}


double GMM::_PDF(Mat datum, Mat rerata, Mat covariance){
	if(!determinant(covariance)){
		cout << "covariance matrix singular !!!" << endl;
		return -1.0f;
	}
	double PI_detSIGMA = pow(pow(M_PI, _dimensi) * determinant(covariance), 0.5);

	// cout << PI_detSIGMA << endl;
	Mat mPANGKAT = -0.5*(datum - rerata).t() * covariance.inv() * (datum - rerata);
	double PANGKAT = mPANGKAT.at<double>(0,0);

	return pow(M_E, PANGKAT)/PI_detSIGMA;
}

bool GMM::isConvergence(vector<double> bobot_alpha_k, vector<Mat> _miyu, vector<Mat> _sigma, double& before_log){
	double current_log = 0.0;
	for(int n=0; n<nData; n++){
		double jmlPDFxAlpha = 0.0;
		for(int k=0; k<nKluster; k++){
			jmlPDFxAlpha += bobot_alpha_k[k] * _PDF(_data_training[n], _miyu[k], _sigma[k]);
		}
		current_log += log10(jmlPDFxAlpha);
	}

	cout << fabs(current_log - before_log) << endl;

	if(fabs(current_log - before_log) < 1e-8)
		return true;

	before_log = current_log;
	return false;
}

void GMM::train(int iterasi, bool saveit){
	double log_likely_hood = 0.0;
	for(int i=0; i<iterasi; i++){
		cout << "iterasi " << i << endl;
		if(isConvergence(_alpha_k, mu_k, Sigma_k, log_likely_hood))
			break;

		double* Nk = new double[nKluster];
		for(int k=0; k<nKluster; k++){
			Nk[k] = 0;
			double* _pix_wi_k = _w_i_k[k].ptr<double>(0);
			Mat wX = Mat::zeros(_dimensi, 1, CV_64FC1);
			Mat gtSigma = Mat::zeros(Sigma_k[k].size(), CV_64FC1);

			for(int n=0; n<nData; n++){
				Nk[k] += _pix_wi_k[n];
				wX += _pix_wi_k[n] * _data_training[n];
			}
			_alpha_k[k] = Nk[k]/nData;
			mu_k[k] = wX/Nk[k];

			for(int n=0; n<nData; n++){
				gtSigma += _pix_wi_k[n] * (_data_training[n] - mu_k[k]) * (_data_training[n] - mu_k[k]).t();
			}
			Sigma_k[k] = gtSigma;

			double sigma_alphaXpdf = 0.0;
			for(int l=0; l<nKluster; l++){
				double* _pix_wi_k = _w_i_k[l].ptr<double>(0);
				for(int n=0; n<nData; n++){
					_pix_wi_k[l] = _alpha_k[l] * _PDF(_data_training[n], mu_k[l], Sigma_k[l]);
					sigma_alphaXpdf += _pix_wi_k[n];
				}
			}
			_w_i_k[k] /= sigma_alphaXpdf;
		}
	}


	FileStorage fs(saveToYAML, FileStorage::WRITE);
	fs << "nKluster" << nKluster;
	fs << "_dimensi" << _dimensi;
	for(int k=0; k<nKluster;k++){
		stringstream ss;
		ss << k;
		fs << "Mean_"+ss.str() << mu_k[k];
		fs << "Cov_"+ss.str() << Sigma_k[k];
		fs << "Alpha_"+ss.str() << _alpha_k[k];
	}
	fs.release();

	cout << "Training done !!" << endl;
	if(saveit){
		cout << "Saving....." << endl;
		save2LUTyaml();
		cout << "done! saved!" << endl;
	}
}

void GMM::save2LUTyaml(){
	loadConfig(saveToYAML);
	for(int b=0; b<256; b++){
		for(int g=0; g<256; g++){
			for(int r=0; r<256; r++){
				double B = double(b) / 4.0;
				double G = double(g) / 4.0;
				double R = double(r) / 4.0;

				Mat _pixel = (Mat_<double>(3,1) << B, G, R);
				int predicted = predict(_pixel);

				int Bind = b >> 2;
				int Gind = g >> 2;
				int Rind = r >> 2;

				canvasYAML.at<int>(((Bind << 6) << 6) + (Gind << 6) + Rind) = predicted;
			}
		}
	}

	cout << "canvas filled" << endl;
	FileStorage fs2(lutYAML, FileStorage::WRITE);
	fs2 << "lutYAML" << canvasYAML;
	fs2.release();
}

void GMM::loadConfig(string configYAML){
	FileStorage fs(configYAML, FileStorage::READ);
	fs["nKluster"] >> nKluster;

	fs["_dimensi"] >> _dimensi;

	mu_k.resize(nKluster);
	Sigma_k.resize(nKluster);
	_alpha_k.resize(nKluster);

	for(int k=0; k<nKluster; k++){
		stringstream _ss;
		_ss << k;

		fs["Mean_" + _ss.str()] >> mu_k[k];
		fs["Cov_" + _ss.str()] >> Sigma_k[k];
		fs["Alpha_" + _ss.str()] >> _alpha_k[k];
	}

	fs.release();
}

int GMM::predict(const Mat& raw_pixel){
	int kluster = 0;
	Mat _x = raw_pixel;
	double _maksimum = 0.0;
	for(int k=0; k<nKluster; k++){
		double alphaXpdf = _alpha_k[k] * _PDF(_x, mu_k[k], Sigma_k[k]);

		if(alphaXpdf > _maksimum){
			_maksimum = alphaXpdf;
			kluster = k;
		}
	}

	return kluster;
}
