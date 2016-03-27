#include <iostream>
#include <fstream>
using namespace std;

class DeepFeatures : public Classifier
{
public:
  DeepFeatures(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // Nearest neighbor training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  {
	//to delete older model files
	string del = "rm -f deep_model_svm.txt";
	system(del.c_str());

	//storing all folder names for category id used for svm format conversion later
	string folders[filenames.size()];
	int count = 0;
	for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
		folders[count++] = c_iter->first;

	string filename, command;
	int start, len;

	//creating models
	for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
	cout << "Processing " << c_iter->first << endl;
	CImg<double> class_vectors(size*size*3, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++)
	  class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));

	for(int i=0; i<c_iter->second.size(); i++){
		start = 7+c_iter->first.length();
		len = c_iter->second[i].length()-start-4;
		filename = "deep_model_" + c_iter->first + "_" + c_iter->second[i].substr(start,len)+".txt";
		cout << filename << endl;
		freopen(filename.c_str(),"w",stdout);
		command = "./overfeat/bin/linux_64/overfeat -L 12 " + c_iter->second[i];
		system(command.c_str());
		fclose(stdout);
		freopen("/dev/tty", "a", stdout);
	}
	cout << endl;
      }

    //modifying models for svm
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {


	ifstream myReadFile;
	ofstream myWriteFile;
	int no_of_features, feature_height, feature_width, next, feature_no = 1, image_no = 0;

	for(int file_no = 0; file_no < c_iter->second.size(); file_no++) { //50
		start = 7 + c_iter->first.length();
		len = c_iter->second[file_no].length()-start-4;
		filename = "deep_model_" + c_iter->first + "_" + c_iter->second[file_no].substr(start,len)+"_svm.txt";
		myReadFile.open(("deep_model_" + c_iter->first + "_" + c_iter->second[file_no].substr(start,len) + ".txt").c_str());
		myWriteFile.open("deep_model_svm.txt", ofstream::out | ofstream::app);
		char output[20];
		if (myReadFile.is_open()) {
			myReadFile >> no_of_features;
			myReadFile >> feature_height;
			myReadFile >> feature_width;
			next = feature_height * feature_width;

			//calculate category number of food product
			for(int count=0;count<filenames.size();count++)
				if(folders[count].compare(c_iter->first) == 0) {
					image_no = count;
					break;
				}

			myWriteFile << image_no << ' ';
			image_no++;
			int i = 0;
			feature_no = 1;
			while (!myReadFile.eof()) {
				myReadFile >> output;
				myWriteFile << feature_no << ':';
				feature_no++;
				myWriteFile << output << ' ';
				for(int skip = 0; skip < next; skip++)
					myReadFile >> output;
			}
			//write the filename at the end of the line
			myWriteFile << ' ' << '#' << ' ' << c_iter->second[file_no] << '\n';
		}
		myReadFile.close();
		myWriteFile.close();
      }
      }
    //deleting unnecessary model files
//    string del = "rm -f deep_model_*_.txt";
//	system(del.c_str());
  }

  virtual string classify(const string &filename)
  {
    CImg<double> test_image = extract_features(filename);
	      
    // figure nearest neighbor
    pair<string, double> best("", 10e100);
    double this_cost;
    for(int c=0; c<class_list.size(); c++)
      for(int row=0; row<models[ class_list[c] ].height(); row++)
	if((this_cost = (test_image - models[ class_list[c] ].get_row(row)).magnitude()) < best.second)
	  best = make_pair(class_list[c], this_cost);

    return best.first;
  }

  virtual void load_model()
  {
    for(int c=0; c < class_list.size(); c++)
      models[class_list[c] ] = (CImg<double>(("deep_model_" + class_list[c] + ".txt").c_str()));
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
      return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }

  static const int size=20;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
