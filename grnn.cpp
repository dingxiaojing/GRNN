#include <math.h>
#include "grnn.h"

void Grnn::train(double *X,double *Y,int num_train,int X_dim)
{
	X_sample=X;
	Y_sample=Y;
	num_sample=num_train;
	X_dimension=X_dim;
}

double Grnn::guess(double *X)
{
	double *w=new double[num_sample];
	for(int i=0;i<num_sample;i++)
	{
		double d=0;
		for(int j=0;j<X_dimension;j++)
		{
			d+=pow(X_sample[i*X_dimension+j]-X[j],2);
		}
		w[i]=exp(-d/(2*pow(sigma,2)));
	}
	double numerator=0;
	double denominator=0;
	for(int i=0;i<num_sample;i++)
	{
		denominator+=w[i];
		numerator+=w[i]*Y_sample[i];
	}
	delete [] w;
	double res=numerator/denominator;
	return res;
}

void Grnn::smooth(float sig)
{
	sigma=sig;
}


Grnn::Grnn()
{
	sigma=1;
	X_dimension=1;
}
Grnn::~Grnn()
{

}