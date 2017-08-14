#include <iostream>
#include <fstream>
#include <string>
#include "grnn.h"
using namespace std;

void readcsv(double data[],string path)
{
	ifstream file;
	file.open(path);
	int idx=0;
	string strline;
	string all;
	const char *split=",";
	char *p;
	while(file.good())
	{
		getline(file,strline,'\n');
		if(strline.length()!=0)
			strline+=",";
		all+=strline;
	}
	//cout<<all<<endl;
	char *cstr=new char[all.length()+1];
	strcpy(cstr,all.c_str());
	p=strtok(cstr,split);
	while(p!=NULL)
	{
		data[idx]=atof(p);
		idx++;
		p=strtok(NULL,split);
	}
	delete [] cstr;
	file.close();
}

double eval_sigma(double *X_sample,double *Y_sample,float sigma,int num_sample,int X_dimension)
{
	double *X_train=new double[(num_sample-1)*X_dimension];
	double *Y_train=new double[num_sample-1];
	double *X_test=new double[X_dimension];
	double Y_test;
	double mse=0.0;
	for(int i=0;i<num_sample;i++)
	{
		Y_test=Y_sample[i];
		for (int k=0;k<X_dimension;k++)
		{
			X_test[k]=X_sample[i*X_dimension+k];
		}
		for(int j=0;j<i;j++)
		{
			for(int k=0;k<X_dimension;k++)
			{
				X_train[j*X_dimension+k]=X_sample[j*X_dimension+k];
			}
			Y_train[j]=Y_sample[j];
		}
		for(int j=i+1;j<num_sample;j++)
		{
			for(int k=0;k<X_dimension;k++)
			{
				X_train[(j-1)*X_dimension+k]=X_sample[j*X_dimension+k];
			}
			Y_train[j-1]=Y_sample[j];
		}

		Grnn g2;
		g2.train(X_train,Y_train,num_sample-1,X_dimension);
		g2.smooth(sigma);
		mse+=pow(g2.guess(X_test)-Y_test,2);
	}
	mse/=num_sample;
	delete [] X_train;
	delete [] X_test;
	delete [] Y_train;
	return mse;
}

void grnn_predict(double *X,double *Y,double *predict,double *result,int num_sample,int X_dim,int num_pred)
{
	//choose the best sigma mininizing the mse using holdout method.
	float s=0.03;
	float best_sigma=0.03;
	double min_err=eval_sigma(X,Y,0.03,num_sample,X_dim);
	while(s<0.2)
	{
		s+=0.01;
		double err=eval_sigma(X,Y,s,num_sample,X_dim);
		//cout<<"sigma="<<s<<" "<<"mse="<<err<<endl;
		if(err<min_err)
		{
			best_sigma=s;
			min_err=err;
		}
	}
	//cout<<"best sigma: "<<best_sigma<<endl;
	//make the prediction
	Grnn g;
	double res;
	double *test=new double[X_dim];	
	g.train(X,Y,num_sample,X_dim);			//set the grnn model samples
	g.smooth(best_sigma);	//set the smooth parameter
	for(int i=0;i<num_pred;i++)
	{
		//for each line of prediction,make guess with grnn
		
		for(int j=0;j<X_dim;j++)
		{
			test[j]=predict[i*X_dim+j];
		}
		//for(int i=0; i<X_dim;i++)
		//{
		//	cout<<test[i]<<" ";
		//}
		res=g.guess(test);
		result[i]=res;
		//cout<<"predict="<<res<<endl;
	}
	delete [] test;
}

int main()
{
	
	//read data from csv file
	double X[40][5]={0.0};
	double Y[40]={0.0};
	double predict[10][5]={0.0};
	readcsv(&X[0][0],"x.csv");
	readcsv(Y,"y.csv");
	readcsv(&predict[0][0],"predict.csv");

	int num_sample=sizeof(Y)/sizeof(Y[0]);	//the number of samples
	int X_dim=sizeof(X[0])/sizeof(X[0][0]);   //the dimension of input variable
	int num_pred=sizeof(predict)/sizeof(predict[0]);	//number of prediction
	//cout<<X_dim<<" "<<num_sample<<" "<<num_pred<<endl;

	double *result=new double[num_pred];
	grnn_predict(&X[0][0],Y,&predict[0][0],result,num_sample,X_dim,num_pred);
	for(int i=0;i<num_pred;i++)
	{
		cout<<result[i]<<endl;
	}
	delete [] result;
	while(1);
	return 0;
}