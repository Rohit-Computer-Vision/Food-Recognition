#include<fstream>
using namespace std;
class svm : public Classifier
{
public:
  svm(const vector<string> &_class_list) : Classifier(_class_list) {}
  ofstream myfile;
  
  // Nearest neighbor training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  {
  	//cout<<filenames<<endl;
    myfile.open ("dat.txt");
    int p=0;
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {

        p=p+1;
		cout << "Processing " << c_iter->first << endl;
		CImg<double> class_vectors(40*40*3, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++){
		class_vectors = extract_features(c_iter->second[i].c_str());

		int actual_size = class_vectors.width() * class_vectors.height();
		//cout<<actual_size<<endl;
	        int k=0;
        	myfile << p << " ";
		for( int j = 0; j < actual_size; j++){
			k=k+1;
			myfile << k << ":" << class_vectors[j] << " " ;
		}
	
		myfile <<"\n";
	}
	
      }
	myfile.close();
  
  system(" ./svm_multiclass_learn -c 1.0 dat.txt");
  }

int dir_count=1;
int count = 1;

virtual string classify(const string &filename)
{	

	ofstream myfile;
	myfile.open("testing.txt");
	CImg<double> test_image = extract_features(filename);
  	if (count == 11){
  		dir_count++;
  		count = 1;
  		myfile << dir_count<<" ";
  	}
  	else{
  		myfile << dir_count<<" ";
  		count++;
    }
     
    
	for( int j = 0; j < test_image.size(); j++)
			myfile << j+1 << ":"<< test_image[j] << " " ;
	
	myfile <<"\n";
	myfile.close();
    system("./svm_multiclass_classify testing.txt svm_struct_model");
	
    std::ifstream input("svm_predictions");
    int x;
    input>>x;
    // cout<<x;
    return class_list[x-1];
	
}


  virtual void load_model()
  {
  
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
      return (CImg<double>(filename.c_str())).resize(40,40,1,3).unroll('x');
    }

  //static const int size=20;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
