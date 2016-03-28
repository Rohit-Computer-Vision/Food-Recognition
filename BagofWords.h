class BagofWords : public Classifier
{
public:
  BagofWords(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // SIFT Bag of Words training. 
  // Get a set of SIFT features for each image, cluster them using k-means
  // represent each image as a vector with counts of the clustered visual words
  virtual void train(const Dataset &filenames) 
  {
    long total_sift = 0;
    remove( "sift.csv" ); 
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
	cout << "Extracting SIFT " << c_iter->first << endl;
	
	// extract and save SIFT vector from each file
	for(int i=0; i<c_iter->second.size(); i++)
	  total_sift = total_sift + sift_extract_features(c_iter->second[i].c_str());
	
      }

  // here do the clustering down to a smaller set of features
	float kmeans_matrix[k_count][sift_depth];
	for(int kx=0; kx<k_count; kx++)
	{
		for(int sx=0; sx<sift_depth; sx++)
		{
		// initialize a cluster center
  		// create random values for cluster centers
			kmeans_matrix[kx][sx] = 0;
		}
	}

  // read sift table into matrix
    float sift_matrix[total_sift][sift_depth];
    std::ifstream  data("sift.csv");

    long lineX = 0;
    long fieldX = 0;
    std::string line;
    while(std::getline(data,line))
    {
        std::stringstream  lineStream(line);
        std::string        cell;
        while(std::getline(lineStream,cell,','))
        {
		if (fieldX > 1)
		{
			sift_matrix[lineX][fieldX-2] = atoi(cell.c_str());
		}
        }
	fieldX = 0;
        lineX++;
    }

  // iterate clusters toward convergence
	for(int i=0; i<5; i++)
	{
		// assign every vector to a cluster
		for (int tx=0; tx < total_sift; tx++)
		{
			// find closest cluster center
		}

		// calculate new mean for every cluster 
		for (int cx=0; cx < k_count; cx++)
		{
			// calculate new mean
		}
	}

  // write out finished clusters for use by classification
  // not yet implemented


    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
	cout << "Processing " << c_iter->first << endl;
	CImg<double> class_vectors(size*size*3, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++)
	  class_vectors = class_vectors.draw_image(0, i, 0, 0, train_extract_features(c_iter->second[i].c_str()));
	
	class_vectors.save_png(("bag_model." + c_iter->first + ".png").c_str());
      }

  }

  virtual string classify(const string &filename)
  {
    CImg<double> test_image = test_extract_features(filename);
    cout << "SIFT and k-means classifying not properly implemented" << endl;
	      
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
      models[class_list[c] ] = (CImg<double>(("bag_model." + class_list[c] + ".png").c_str()));
  }
protected:

  // retrieve SIFT descriptors, cluster, and return a vector of counts
  long sift_extract_features(const string &filename)
    {
      long feature_counter = 0;
      CImg<double> original_image(filename.c_str());
      original_image.resize(500,500,1,3); // needed?
      CImg<double> original_gray = original_image.get_RGBtoHSI().get_channel(2);

	vector<SiftDescriptor> query_descriptors = Sift::compute_sift(original_gray);

  // need to write every descriptor to a file for later processing

      ofstream SIFTfile;
      SIFTfile.open ("sift.csv", std::ios_base::app);

	for(int j = 0; j < query_descriptors.size(); j++)
	{
      		SIFTfile << filename.c_str() << ',' << j;
		for(int k = 0; k < query_descriptors[j].descriptor.size();k++)
		{
			SIFTfile << ',' << query_descriptors[j].descriptor[k];
			feature_counter++;
		}
      		SIFTfile << "\n";
	}

      SIFTfile.close();

      return(feature_counter);
    }

  // retrieve SIFT descriptors, cluster, and return a vector of counts
  CImg<double> train_extract_features(const string &filename)
    {

      CImg<double> original_image(filename.c_str());
      original_image.resize(500,500,1,3); // needed?
      CImg<double> original_gray = original_image.get_RGBtoHSI().get_channel(2);

	vector<SiftDescriptor> query_descriptors = Sift::compute_sift(original_gray);

      return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }


  // retrieve SIFT descriptors, cluster, and return a vector of counts
  CImg<double> test_extract_features(const string &filename)
    {
      CImg<double> original_image(filename.c_str());
      original_image.resize(500,500,1,3); // needed?
      CImg<double> original_gray = original_image.get_RGBtoHSI().get_channel(2);

	vector<SiftDescriptor> query_descriptors = Sift::compute_sift(original_gray);

  // k-means to cluster descriptors
  // calculate a vector of visual word counts, that's what will be returned

      return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }


  static const int k_count=200;  // subsampled image resolution
  static const int sift_depth=200;  // subsampled image resolution
  static const int size=20;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
