#include "GMM_EM.hpp"

GMM::GMM(int jmlKluster, vector<Mat> dataset, vector<Mat> initialR)
	:nKluster(jmlKluster), nData(dataset.size())
	, _data_training(dataset)
{
	_dimensi = initialR[0].rows;
	mu_k.resize(nKluster);
	Sigma_k.resize(nKluster);
	_alpha_k.resize(nKluster);

	_w_i_k.resize(nKluster);
	_pdf_k.resize(nKluster);

	for(int k=0; k<jmlKluster; k++){
		_w_i_k[k] = Mat::zeros(1, nData, CV_64FC1);
		_pdf_k[k] = Mat::zeros(1, nData, CV_64FC1);

		_alpha_k[k] = 1.0/jmlKluster;
	}

	mu_k = initialR;

	for(int k=0; k<nKluster; k++){
		Sigma_k[k] = Mat::zeros(initialR[0].rows, initialR[0].rows, CV_64FC1);
		for(int n=0; n<dataset.size(); n++){
			Sigma_k[k] += (dataset[n] - mu_k[k]) * (dataset[n] - mu_k[k]).t();
		}
		Sigma_k[k] /= dataset.size();
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

void GMM::train(int iterasi, string saveToYAML){//, vector<string> name_kluster){
	double log_likely_hood = 0.0;
	for(int i=0; i<iterasi; i++){
		cout << "step " << i << ": ";
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
}

GMM::GMM(){
	// cout << "object created" << endl;
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

	// cout << Sigma_k[0] << endl;
	fs.release();
}

int GMM::predict(const Mat& raw_pixel){
	// Mat _x = raw_pixel;
	// double sumAlphaXpdf = 0.0;
	// Mat xToKluster = Mat::zeros(1, nKluster, CV_64FC1);
	
	// double* ptr_xToKluster = xToKluster.ptr<double>(0);

	// for(int k=0; k<nKluster; k++){
	// 	sumAlphaXpdf += _alpha_k[k]*_PDF(_x, mu_k[k], Sigma_k[k]);
	// 	ptr_xToKluster[k] = _alpha_k[k]*_PDF(_x, mu_k[k], Sigma_k[k]);
	// }

	// double _max = 0.0;
	// int kluster = -1;
	// for(int k=0; k<nKluster; k++){
	// 	double _batas = ptr_xToKluster[k] * log(ptr_xToKluster[k])/sumAlphaXpdf;
	// 	if(_max < _batas){
	// 		_max = _batas;
	// 		kluster = k;
	// 	}
	// }

	int kluster = 0;
	Mat _x = raw_pixel;
	double _maksimum = 0.0;
	for(int k=0; k<nKluster; k++){
		double alphaXpdf = _alpha_k[k] * _PDF(_x, mu_k[k], Sigma_k[k]);
		// cout << _PDF(_x, mu_k[k], Sigma_k[k]) << endl;

		if(alphaXpdf > _maksimum){
			_maksimum = alphaXpdf;
			kluster = k;
		}
	}

	// cout << kluster << endl;
	return kluster;
}


Mat gambar;
Mat gambarClone;
Point titikStart;
bool afterDownBeforeUp = false;
Rect rectROI;

static void onMouse(int event, int x, int y, int, void*){
    int xrs, yrs, lx, ly;

    if(afterDownBeforeUp){
        gambar = gambarClone.clone();
        xrs = min(titikStart.x, x);
        yrs = min(titikStart.y, y);
        lx = max(titikStart.x, x) - min(titikStart.x, x);
        ly = max(titikStart.y, y) - min(titikStart.y, y);
        rectROI = Rect(xrs, yrs, lx+1, ly+1);

        rectangle(gambar, rectROI,Scalar(255, 0, 0), 1);
    }
    if(event == EVENT_LBUTTONDOWN){
        titikStart = Point(x,y);
        rectROI = Rect(x,y,0,0);
        afterDownBeforeUp = true;

    }else if(event == EVENT_LBUTTONUP){
        Mat roi(gambarClone.clone(), rectROI);
        imshow("roi", roi);

        afterDownBeforeUp = false;
    }
}


vector<Mat> masukanMatrix(Mat gambar, Rect kotak){
    int xrs, yrs, xrf, yrf;
    xrs = kotak.x;
    yrs = kotak.y;
    xrf = xrs + kotak.width;
    yrf = yrs + kotak.height;

    vector<Mat> RGB;
    for(int xx=xrs+1; xx<xrf; xx++){
        for(int yy=yrs+1; yy<yrf; yy++){
            Vec3b pixel = gambar.at<Vec3b>(yy,xx);

            double R = (double)pixel[2];
            double G = (double)pixel[1];
            double B = (double)pixel[0];

            RGB.push_back((Mat_<double>(3,1) << B, G, R));
        }
    }

    return RGB;
}


int main(){
	VideoCapture vc("/home/udiro/Music/gawangPkkh.mp4");

	vector<Mat> _datasetWarna;

	std::vector<string> namaKluster;
	namaKluster.push_back("hijau");
	namaKluster.push_back("putih");
	namaKluster.push_back("other");

	int state = GMM::STATE_PREDICT;

	if(! vc.isOpened())
		return -1;
	while(true){
		Mat frame;		
		vc.read(frame);

		resize(frame, frame, Size(), 640.0/(double)frame.cols, 480.0/(double)frame.rows);

		namedWindow("frame", CV_WINDOW_NORMAL);
		imshow("frame", frame);
		int key = waitKey(10);
		if((char) key == 'k'){
			namedWindow("kalibrasiFrame", CV_WINDOW_NORMAL);
			setMouseCallback("kalibrasiFrame", onMouse);
			
			gambar = frame;
			gambarClone = gambar.clone();

			while(true){
				int inkey = waitKey(10);
				imshow("kalibrasiFrame", gambar);

				if((rectROI.width != 0) || (rectROI.height != 0) ){
	                bool kalib = true;
	                vector<Mat> dataTemp;

	                if((char)inkey == 's'){
                    	dataTemp = masukanMatrix(gambar, rectROI);
                    	cout << "data saved !!!" << endl;
	                }

		            if(dataTemp.size()){
		            	if(_datasetWarna.size() > 3000){
							cout << "Dataset udah banyak brooo" << endl;
						 }else{
						 	// int tambah = (3000 - _datasetWarna.size()) > dataTemp.size() ? dataTemp.size() : (3000 - _datasetWarna.size());
						 	int tambah = dataTemp.size() < 200 ? dataTemp.size()-1 : 200;
		            		_datasetWarna.insert(_datasetWarna.end(), dataTemp.begin(), dataTemp.begin() + tambah);
		            	 }
		            	cout << _datasetWarna.size() << endl;
		            }
            	}

				if((char) inkey == 'c'){
					destroyAllWindows();
					break;
				}
			}
		}

		if(state == GMM::STATE_COLLECT){
			if(! _datasetWarna.size()){
				cout << "Harus ada dataset BROO" << endl;
			}else if(_datasetWarna.size() > 3000){
				int banyakKluster = 3;
				vector<Mat> awal;
				awal.push_back((Mat_<double>(3,1) << 0.0, 255.0, 0.0));
				awal.push_back((Mat_<double>(3,1) << 255.0 , 255.0, 255.0));
				awal.push_back((Mat_<double>(3,1) << 10, 20, 30));
			

				GMM _gaussian(banyakKluster, _datasetWarna, awal);
				_gaussian.train(2000, "RGB.yaml");
				state = GMM::STATE_PREDICT;
			}
		}

		if(state == GMM::STATE_PREDICT){
			Mat blank = Mat::zeros(frame.size(), CV_8UC3);

			GMM _gaus_predict;
			_gaus_predict.loadConfig("RGB.yaml");
			for(int r=0;r<frame.rows; r++){
				Vec3b* ptr_ = frame.ptr<Vec3b>(r);
				Vec3b* _ptr_blank = blank.ptr<Vec3b>(r);

				for(int c=0; c<frame.cols; c++){					
					double B = (double)ptr_[c][2];
					double G = (double)ptr_[c][1];
					double R = (double)ptr_[c][0];

					// cout << B << " " << G << " " << R << endl << endl;

					Mat _pixel = (Mat_<double>(3,1) << B, G, R);
					// cout << _pixel << endl;

					/* MASALAH DI SINI*/
					int predicted = _gaus_predict.predict(_pixel);
					/**/
					// cout << predicted << endl;
					switch(predicted){
						case 0:
							// cout << "0 is " << Point(c,r) << endl;
							_ptr_blank[c] = Vec3b(0, 0, 255);
							break;
						case 1: 
							// cout << "1 is " << Point(c,r) << endl;
							_ptr_blank[c] = Vec3b(0, 255, 0);
							break;
						case 2: 
							// cout << "2 is " << Point(c,r) << endl;
							_ptr_blank[c] = Vec3b(255, 0, 0);
							break;
					}
				}
			}
			namedWindow("labeled", CV_WINDOW_NORMAL);
			imshow("labeled", blank);
		}
		if(key == 27)
			break;
	}
}