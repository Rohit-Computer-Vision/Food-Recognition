class BagofWords : public Classifier
{
public:
  BagofWords(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // SIFT Bag of Words training. 
  // Get a set of SIFT features for each image, cluster them using k-means
  // represent each image as a vector with counts of the clustered visual words
  virtual void train(const Dataset &filenames) 
  {
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
	cout << "Processing " << c_iter->first << endl;
	CImg<double> class_vectors(size*size*3, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++)
	  class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
	
	class_vectors.save_png(("nn_model." + c_iter->first + ".png").c_str());
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
      models[class_list[c] ] = (CImg<double>(("nn_model." + class_list[c] + ".png").c_str()));
  }
protected:
  // retrieve SIFT descriptors, cluster, and return a vector of counts
  CImg<double> extract_features(const string &filename)
    {

      CImg<double> original_image(filename.c_str());
      original_image.resize(500,500,1,3); // needed?
      CImg<double> original_gray = original_image.get_RGBtoHSI().get_channel(2);

	vector<SiftDescriptor> query_descriptors = Sift::compute_sift(original_gray);
	cout << "number of sift descriptors for query image: " << query_descriptors.size() << endl;

  // k-means to cluster descriptors
  // calculate a vector of visual word counts, that's what will be returned

      return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }

  static const int size=20;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
