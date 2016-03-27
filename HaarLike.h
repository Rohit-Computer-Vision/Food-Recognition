class HaarLike : public Classifier
{
public:
  HaarLike(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // Haar-like feature extraction and Viola/Jones training. 
  // Read in all the images, resize them to a common size & convert to greyscale
  // Calculate differences of summed pixel values in selected rectangles as features

  virtual void train(const Dataset &filenames) 
  {
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
	cout << "Processing " << c_iter->first << endl;
	CImg<double> class_vectors(size*size, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++)
	  class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
	
	class_vectors.save_png(("haar_model." + c_iter->first + ".png").c_str());
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
      models[class_list[c] ] = (CImg<double>(("haar_model." + class_list[c] + ".png").c_str()));
  }
protected:
  // extract features from an image.
  // vector returned is sums and differences of calculated regions
  CImg<double> extract_features(const string &filename)
    {

  // start with grayscale version of original image
    
      CImg<double> original_image(filename.c_str());
      original_image.resize(haarImageSize,haarImageSize,1,3);
      CImg<double> features_matrix(size, size,1, 1);

      CImg<double> original_gray = original_image.get_RGBtoHSI().get_channel(2);

  // allocate and populate integral image
	CImg<double> integral_image(haarImageSize, haarImageSize, 1, 1);

	for (int colx = 0; colx<haarImageSize ; colx++)          
	{
		for (int rowx = 0; rowx < haarImageSize; rowx++)
		{
			integral_image(colx,rowx) = original_gray(colx,rowx);
   			if (colx > 0) integral_image(colx,rowx) = integral_image(colx,rowx) + integral_image(colx-1,rowx);
   			if (rowx > 0) integral_image(colx,rowx) = integral_image(colx,rowx) + integral_image(colx,rowx-1);
   			if ((rowx > 0) && (colx > 0)) integral_image(colx,rowx) = integral_image(colx,rowx) - integral_image(colx-1,rowx-1);
		}
	}

  // calculate features based on pixel sums/differences in selected rectangles

        int thisUp = 0;
	int thisLeft = 0;
        double plusValue = 0.0;
        double minusValue = 0.0;
        double featureValue = 0.0;
        int featureHeight = 50;
        int featureWidth = 50;
	int rowIndex = 0;
	int colIndex = 0;

	for (int colx = 0; colx<size; colx++)          
	{
		for (int rowx = 0; rowx < size; rowx++)
		{
			features_matrix(rowx,colx) = 0.0;
		}
	}

	for (int runX = 0; runX < 38; runX++)
	{


	for (int colx = 1; colx<((haarImageSize/featureWidth)-1) ; colx++)          
	{
		for (int rowx = 1; rowx < ((haarImageSize/featureHeight)-1); rowx++)
		{
			thisLeft = colx * featureWidth * 2;
			thisUp = rowx * featureHeight;
			plusValue = integral_image(thisLeft,thisUp) + integral_image(thisLeft+featureWidth,thisUp+featureHeight)
			- integral_image(thisLeft+featureWidth,thisUp) - integral_image(thisLeft,thisUp+featureHeight);
			minusValue = integral_image(thisLeft,thisUp+featureHeight) + integral_image(thisLeft+featureWidth,thisUp+(2*featureHeight))
			- integral_image(thisLeft+featureWidth,thisUp+featureHeight) - integral_image(thisLeft,thisUp+(2*featureHeight));
			featureValue = plusValue - minusValue;
			featureValue = featureValue + 50000;
			featureValue = featureValue / (100000/255);
			features_matrix(colIndex,rowIndex) = featureValue; //35
			colIndex++;
			if (colIndex == size)
				{
					colIndex = 0;
					rowIndex++;
				}
		}
	}

	featureHeight = featureHeight - 1;
	featureWidth = featureWidth - 1;

	}
  //	cout << rowIndex << " " << colIndex << " "  << endl;
      return (features_matrix.unroll('x'));
    }

  static const int size=50;  // total number of features (adjust if image or feature size changes)
  static const int haarImageSize=250;
  map<string, CImg<double> > models; // trained models
};

