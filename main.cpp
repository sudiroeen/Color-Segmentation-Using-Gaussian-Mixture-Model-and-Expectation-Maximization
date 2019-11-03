/*
Copyright by:
  Sudiro
    [at] SudiroEEN@gmail.com
*/

#include "GMM_EM.hpp"

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

// #define LOAD_DATASET
#define MANUAL_DATASET

int main(){
	vector<Mat> _datasetWarna;

	std::vector<string> namaKluster;

	bool saveit;
#ifdef SAVE
	saveit = true;
#else
	saveit = false;
#endif

	int nklaster = 3;
	for(int s=0; s<nklaster; s++){
		stringstream ss;
		ss << "class_" << s;
		namaKluster.push_back(ss.str());
	}

	int state = GMM::STATE_TRAIN;

	Mat foto = imread("jalan.png");

	while(true){
		Mat frame = foto.clone();

		resize(frame, frame, Size(), 640.0/(double)frame.cols, 480.0/(double)frame.rows);

		namedWindow("frame", CV_WINDOW_NORMAL);
		imshow("frame", frame);
		int key = waitKey(10);

#ifdef MANUAL_DATASET
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
	            		_datasetWarna.insert(_datasetWarna.end(), dataTemp.begin(), dataTemp.end());
		            }
            	}

				if((char) inkey == 'c'){
					destroyAllWindows();
					break;
				}
			}
		}
#endif

		if(state == GMM::STATE_TRAIN){
#ifdef LOAD_DATASET
			destroyAllWindows();
			FileStorage fsdata("dataset.yaml", FileStorage::READ);
			int datasetSize;
			fsdata ["datasetSize"] >> datasetSize;
			_datasetWarna.resize(datasetSize);
			for(int d=0; d<datasetSize; d++){
				stringstream ss; ss << d;
				fsdata ["Mat_"+ss.str()] >> _datasetWarna[d];
			}
			fsdata.release();			
#endif
			if(! _datasetWarna.size()){
				cout << "Harus ada dataset BROO" << endl;
			}else if(_datasetWarna.size() > 3000){
				int banyakKluster = namaKluster.size();
				vector<Mat> awal;
				
				Mat m(3,1, CV_64FC1);
				double mean = 127.5;
				double stddev = 100.0;
				for(int s=0; s< namaKluster.size(); s++){
					randn(m, Scalar(mean), Scalar(stddev));
					awal.emplace_back(m);
				}

#ifdef MANUAL_DATASET
				FileStorage fsdata("dataset.yaml", FileStorage::WRITE);
				fsdata << "datasetSize" << (int)_datasetWarna.size();

				for(int d=0; d<_datasetWarna.size(); d++){
					stringstream ss; ss << d;
					fsdata << "Mat_"+ss.str() <<  _datasetWarna[d];
				}
				fsdata.release();
#endif
				string saveToYAML_ = "RGB.yaml";
				GMM gmmAndEM(banyakKluster, _datasetWarna, awal, saveToYAML_, "lutYAML.yaml", 1, (1 << 18));
				gmmAndEM.train(2000, saveit);
				state = GMM::STATE_SEGMENT;
			}
		}

		if(state == GMM::STATE_SEGMENT){
			Mat blank = Mat::zeros(frame.size(), CV_8UC3);
#ifdef SAVE
			Mat MatLUT;
			FileStorage fs("lutYAML.yaml", FileStorage::READ);
			fs["lutYAML"] >> MatLUT;
			fs.release();
#endif
			RNG rng(12345);
			vector<Vec3b> randColor;
			for(int s=0; s<namaKluster.size(); s++){
				randColor.emplace_back(Vec3b(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)));
			}

#ifndef SAVE
			GMM Gpredict;
			Gpredict.loadConfig("RGB.yaml");
#endif

			for(int r=0;r<frame.rows; r++){
				Vec3b* ptr_ = frame.ptr<Vec3b>(r);
				Vec3b* _ptr_blank = blank.ptr<Vec3b>(r);
				for(int c=0; c<frame.cols; c++){
#ifdef SAVE
					int B = ptr_[c][0] >> 2;
					int G = ptr_[c][1] >> 2;
					int R = ptr_[c][2] >> 2;
					int val = MatLUT.at<int>(((B << 6)<< 6) + (G << 6) + R);
#else
					double B = double(ptr_[c][0])/4.0;
					double G = double(ptr_[c][1])/4.0;
					double R = double(ptr_[c][2])/4.0;

					Mat _pixel = (Mat_<double>(3,1) << B, G, R);
					int val = Gpredict.predict(_pixel);
#endif
					_ptr_blank[c] = randColor[val];
				}
			}
			namedWindow("labeled", CV_WINDOW_NORMAL);
			imshow("labeled", blank);
			imwrite("segmented.jpg", blank);
			state = GMM::STATE_OTHER;
		}

		waitKey(1);
		if(key == 27)
			break;
	}
}
