class EigenFood : public Classifier
{
public:
  EigenFood(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // EigenFood training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  {
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
	cout << endl << "Processing " << c_iter->first << endl;
	int cols = size*size*3;
	int rows = filenames.size();
//	cout << endl << "cols=" << cols << ", rows=" << rows << endl;

	CImg<double> class_vectors(cols, rows, 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++)
	  class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));

	//calculating average food vector
	CImg<double> avg_vector(cols,1,1);
	double sum;
	for(int c=0; c<cols; c++) {
		sum = 0.0;
		for(int r=0; r<rows; r++) {
			sum += class_vectors(c,r,0,0);
		}
		avg_vector(c,0,0,0) = sum/rows;
//		cout << endl << "col " << c << ": sum = " << sum << ", avg = " << sum/rows;
	}

	//calculating normalized image vectors
	CImg<double> class_vectors_normalized = class_vectors;
	for(int c=0; c<cols; c++) {
		for(int r=0; r<rows; r++) {
			class_vectors_normalized(c,r,0,0) = class_vectors(c,r,0,0) - avg_vector(c,0,0,0);
		}
	}
//	class_vectors_normalized.save_png(("eigenfood_normalized_" + c_iter->first + ".png").c_str());
//	class_vectors.save_png(("eigenfood_model_" + c_iter->first + ".png").c_str());

	//find transpose of of normalized class vectors
	CImg<double> class_vectors_normalized_transpose = transpose(class_vectors_normalized);

	//covariance
	CImg<double> covariance = class_vectors_normalized * class_vectors_normalized_transpose / rows;
//	covariance.save_png(("eigenfood_covariance_" + c_iter->first + ".png").c_str());
//	cout << endl<< "rows=" << covariance._height << ", cols=" << covariance._width << endl;

	CImg<> U,S,V;
	covariance.SVD(U,S,V);

//	cout << "S: ";
//	for(int r=0; r<25; r++)
//		cout << S(0,r,0,0) << " ";
//	cout << endl;

	CImg<> eigenvectors(10,25,1);
	int select_rows = 10;
	for(int i = 0; i < select_rows; i++)
		for(int j = 0; j < rows; j++)
			eigenvectors(i,j,0,0) = U(i,j,0,0);

//	cout << "eigenvectors: " << eigenvectors._height << "x" << eigenvectors._width << endl;
//	cout << "class_vectors: " << class_vectors._height << "x" << class_vectors._width  << endl;
	CImg<> class_vectors_from_eigen = transpose(eigenvectors) * class_vectors;
//							10x1200	  =	  25x10    *    25x1200
	cout << "class_vectors_from_eigen: " << class_vectors_from_eigen._height << "x" << class_vectors_from_eigen._width << endl;

	CImg<> test = avg_vector;
	for(int i = 0; i < cols; i++){
		test(i,0,0,0) += class_vectors_from_eigen(i,0,0,0);
		cout << i << endl;
		cout << "test : " << test(i,0,0,0) << endl;
		cout << "avg  : " << avg_vector(i,0,0,0) << endl;
		cout << "class: " << class_vectors_from_eigen(i,0,0,0) << endl;
		cout << endl;
	}

//	test.save_png(("test_" + c_iter->first + ".png").c_str());


      }
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
      models[class_list[c] ] = (CImg<double>(("eigenfood_model." + class_list[c] + ".png").c_str()));
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
