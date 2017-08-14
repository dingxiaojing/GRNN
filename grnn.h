class Grnn
{
public:
	void train(double *X,double *Y,int num_train,int X_dim);
	void smooth(float sigma);
	double guess(double *X);
	Grnn();
	~Grnn();

private:
	double *X_sample;
	double *Y_sample;
	int X_dimension;
	int num_sample;
	float sigma;
};