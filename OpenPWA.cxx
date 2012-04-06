#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <iomanip>
#include <cmath>

#include "TComplex.h"
#include "TMath.h"
#include "TRandom3.h"
#include "Math/Minimizer.h"
#include "TStopwatch.h"

#include "OpenPWA.h"
#include "config.h"

using namespace std;

bool resultEval(FitResult elem1, FitResult elem2) 
{ 
  return elem1.s < elem2.s; 
}

OpenPWA::OpenPWA(string configFileName)
{
  logInfo(cout << "Configuring the OpenPWA using config file: " << configFileName << endl << endl);

  fConfig = new Config(configFileName);

  fNevt = fConfig->pInt("nEvent");
  fNMC = fConfig->pInt("nMcEvent");
  fNpar = fConfig->pInt("nParameters");
  fNwave = fConfig->pInt("nPartialWaves");
  fNwave2 = fNwave*fNwave;

  fNbunch = fNevt/4;

  fParaInp = fConfig->pString("ParaInput");

  fMCIntegral = fConfig->pString("MCIntegralFUN");
  fDataFunall = fConfig->pString("DataFUNALL");

  fStepSize = fConfig->pDouble("StepSize");

  fEnable_CL = fConfig->pBool("Enable_CL");
  localSize = fConfig->pInt("LocalSize");
  nPrefold = fConfig->pInt("PreFold");
  //logFile.open("log.txt", std::fstream::out);

  fEnable_FDC = fConfig->pBool("Enable_FDC");

  logInfo(cout << "!!!!The OpenPWA will be initialized using following configuration!!!" << endl << endl);

  logInfo(cout << "Total number of data events: " << fNevt << endl);
  logInfo(cout << "Total number of MC events: " << fNMC << endl << endl);

  logInfo(cout << "Number of partial waves added: " << fNwave << endl);
  logInfo(cout << "Number of amplitude parameters: " << fNpar << endl << endl);

  logInfo(cout << "Initial values of parameter: " << fParaInp << endl);
  logInfo(cout << "FUN values of MC integral: " << fMCIntegral << endl);
  logInfo(cout << "FUN values of every data events: " << fDataFunall << endl << endl << endl);

  if(fEnable_CL)
    {
      logInfo(cout << "OpenCL is enabled! " << endl << endl);
    }
}

OpenPWA::~OpenPWA()
{
  if(fEnable_CL)
    {
      finalizeCL();
    }

  delete fConfig; 

  free(fFunMC); 
  free(fFunall); 

  delete[] fParInfo; 
  delete[] fAmp;
  free(fParVal);  
  free(fParGrad); 
  free(fWaveGrad); 

  free(fPa);  
  free(fPa_grad);  

  fFitResults.clear();  

  cout << "Congratulations! All jobs done!" << endl;

  //logFile.close();
}

void OpenPWA::init()
{
  readMCIntegral();
  readData();
  initializeParameters();

  if(fEnable_CL)
    {
      initializeCL();
    }

  fFitResults.clear();
}

void OpenPWA::initializeParameters()
{
  fParVal = (cl_float *)malloc(fNpar*sizeof(cl_float));
  fParGrad = (double *)malloc(fNpar*sizeof(double));
  fWaveGrad = (double *)malloc(fNwave2*sizeof(double));
  fParInfo = new ParameterInfo[fNpar];
  fAmp = new TComplex[fNwave];

  ifstream fin(fParaInp.c_str());
  char buffer[300];

  if(fEnable_FDC)
    {
      fAmp_FDC = new TComplex[fNpar/2];

      fin.getline(buffer, 300, '\n');
      fin.getline(buffer, 300, '\n');

      for(int i = 0; i < fNpar; i++)
	{
	  fin.getline(buffer, 300, '\n');
	  istringstream stringBuf(buffer);
	  stringBuf >> fParInfo[i].index >> fParVal[i] >> fParInfo[i].step
		    >> fParInfo[i].min >> fParInfo[i].max;
	  
	  if(i%2 == 0)
	    {
	      fParInfo[i].tag = "R";

	      stringstream buff;
	      buff << "R" << int(i/2);
	      buff >> fParInfo[i].name;
	    }
	  else
	    {
	      fParInfo[i].tag = "I";

	      stringstream buff;
	      buff << "I" << int(i/2);
	      buff >> fParInfo[i].name;
	    }
	}
    }
  else
    {
      for(int i = 0; i < fNpar; i++)
	{
	  fin.getline(buffer, 300, '\n');
	  istringstream stringBuf(buffer);
	  stringBuf >> fParInfo[i].index >> fParVal[i] >> fParInfo[i].step
		    >> fParInfo[i].tag >> fParInfo[i].name >> fParInfo[i].min >> fParInfo[i].max;
	}
    }

  fPa = (cl_float *)malloc(fNwave2*sizeof(cl_float));
  fPa_grad = (cl_float *)malloc(fNwave2*sizeof(cl_float));
}

void OpenPWA::readMCIntegral()
{
  fFunMC = (cl_float *)malloc(fNwave2*sizeof(cl_float));

  ifstream fin(fMCIntegral.c_str());
  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  fin >> fFunMC[i*fNwave+j];
	  fFunMC[i*fNwave+j] *= 10;  //*10 is for consistency with fortran code
	}
    }
  fin.close();
}

void OpenPWA::readData()
{
  fFunall = (cl_float *)memalign(16, fNbunch*fNwave2*sizeof(cl_float4));

  ifstream fin(fDataFunall.c_str());
  for(int i = 0; i < fNbunch; i++)
    {
      for(int l = 0; l < 4; l++)
	{
	  for(int j = 0; j < fNwave; j++)
	    {
	      for(int k = 0; k < fNwave; k++)
		{
		  fin >> fFunall[4*i*fNwave2+4*(j*fNwave+k)+l];
		}
	    }
	}
    }
  fin.close();
}

double OpenPWA::Eval_CPU(const double *par)
{
  for(int i = 0; i < fNpar; i++)
    {
      fParVal[i] = par[i];
    }

  return nll_cpu();
}

double OpenPWA::Eval_GPU(const double *par)
{
  for(int i = 0; i < fNpar; i++)
    {
      fParVal[i] = par[i];
    }

  return nll_cl();
}

double OpenPWA::Derivative(const double *par, int icoord)
{
  return fParGrad[icoord];
}


double OpenPWA::nll_cpu()
{
  //load the new parameters
  convertAmplitudeFDC();

  //calculate the MC integral
  double xsec_total = 0;
  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  int idx = i*fNwave + j; 
	  xsec_total += fPa[idx]*fFunMC[idx]; 
	}
    }
  xsec_total = xsec_total/fNMC;

  //Set the gradients of fPa(i,j) to 0 for sum
  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  fWaveGrad[i*fNwave + j] = 0.;
	}
    }

  //calculate the differential x-section
  //single-precision for dcs of a singla event, and double-precision for sum of all events
  float xsec_differ;
  double nll = 0;
  for(int i = 0; i < fNbunch; i++)
    {
      for(int l = 0; l < 4; l++)
	{
	  xsec_differ = 0;
	  for(int j = 0; j < fNwave; j++)
	    {
	      for(int k = 0; k < fNwave; k++)
		{
		  xsec_differ += fPa[j*fNwave+k]*fFunall[4*i*fNwave2+(j*fNwave+k)*4+l];
		}
	    }
	  nll += TMath::Log(xsec_differ);

	  //calc the gradients for Pa(i,j)
	  for(int j = 0; j < fNwave; j++)
	    {
	      for(int k = 0; k < fNwave; k++)
		{
		  fWaveGrad[j*fNwave+k] += fFunall[4*i*fNwave2+(j*fNwave+k)*4+l]/xsec_differ;
		}
	    }
	}
    }
  nll = nll + fNevt*TMath::Log(5./xsec_total);  //*5 is also for legacy code consistency

  double factor_grad = fNevt/xsec_total/fNMC;
  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  fWaveGrad[i*fNwave+j] -= fFunMC[i*fNwave+j]*factor_grad;
	  fWaveGrad[i*fNwave+j] *= -2.;
	}
    }

  for(int i = 0; i < fNpar; i++)
    {
      fParVal[i] += fStepSize;
      convertAmplitudeFDC_grad();

      fParGrad[i] = 0.;
      for(int j = 0; j < fNwave; j++)
	{
	  for(int k = 0; k < fNwave; k++)
	    {
	      int idx = j*fNwave + k;
	      fParGrad[i] += ((fPa_grad[idx] - fPa[idx])/fStepSize*fWaveGrad[idx]);	     
	    }	
	}

      fParVal[i] -= fStepSize;
    }

  // cout << "New Call ! ============================" << endl;
  // for(int i = 0; i < fNwave; i++)
  //   {
  //     for(int  j = 0; j < fNwave; j++)
  // 	{
  // 	  cout << i << "  " << j << "  " << setprecision(20) << fPa[i*fNwave + j] << endl; 
  // 	}
  //   } 
  //cout << setprecision(20) << nll << endl;

  return -2.0*nll;
} 

double OpenPWA::nll_cl()
{
  //TStopwatch timer;

  //load the new parameters
  convertAmplitudeFDC();

  //calculate the MC integral
  double xsec_total = 0;
  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  int idx = i*fNwave + j; 
	  xsec_total += fPa[idx]*fFunMC[idx];  
	}
    }
  xsec_total = xsec_total/fNMC;

  //Enqueue CL calculation
  //timer.Start();
  clEnqueueWriteBuffer(queue, buffer_par, CL_TRUE, 0, fNwave2*sizeof(cl_float), fPa, 0, NULL, NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event_kernel);
  clEnqueueReadBuffer(queue, buffer_nll, CL_FALSE, 0, nBlocks*(1+fNwave2)*sizeof(cl_float), fNll, 1, &event_kernel, &event_nll);
  //clEnqueueReadBuffer(queue, buffer_grad, CL_FALSE, 0, nBlocks*fNwave2*sizeof(cl_float), fGrad, 1, &event_kernel, &event_grad);
  //timer.Stop(); fTime[0] += timer.CpuTime();

  //timer.Start();
  clWaitForEvents(1, &event_nll);
  //clWaitForEvents(1, &event_grad);
  //timer.Stop(); fTime[1] += timer.CpuTime();

  //Calc. negative log likelihood
  //timer.Start();
  double nll = 0.;
  //clWaitForEvents(1, &event_nll);
  for(int i = 0; i < nBlocks; i++)
    {
      //cout << i << "  " << fNll[i] << endl;
      nll += fNll[i*(fNwave2+1)];
      //nll += TMath::Log(fDcs[i]);
    }
  nll = nll + fNevt*TMath::Log(5./xsec_total);
  //timer.Stop(); fTime[2] += timer.CpuTime();

  //Calc. gradients
  //timer.Start();
  double factor_grad = fNevt/xsec_total/fNMC;
  //clWaitForEvents(1, &event_grad);
  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  int idx = i*fNwave + j;

	  fWaveGrad[idx] = 0.;
	  for(int k = 0; k < nBlocks; k++)
	    {
	      //cout << fNll[nBlocks+k*fNwave2+idx] << endl;
	      //fWaveGrad[idx] += fGrad[k*fNwave2+idx];
	      fWaveGrad[idx] += fNll[k*(fNwave2+1)+idx+1];
	    }
	  fWaveGrad[idx] -= (fFunMC[idx]*factor_grad);
	  fWaveGrad[idx] *= -2.;	  
	}
    }
  //timer.Stop(); fTime[3] += timer.CpuTime();

  //timer.Start();
  for(int i = 0; i < fNpar; i++)
    {
      fParVal[i] += fStepSize;
      convertAmplitudeFDC_grad();

      fParGrad[i] = 0.;
      for(int j = 0; j < fNwave; j++)
	{
	  for(int k = 0; k < fNwave; k++)
	    {
	      int idx = j*fNwave + k;
	      fParGrad[i] += ((fPa_grad[idx] - fPa[idx])/fStepSize*fWaveGrad[idx]);	     
	    }	
	}

      fParVal[i] -= fStepSize;
    }
  //timer.Stop(); fTime[4] += timer.CpuTime();

  // cout << "New Call ! ============================" << endl;
  // for(int i = 0; i < fNwave; i++)
  //   {
  //     for(int  j = 0; j < fNwave; j++)
  // 	{
  // 	  cout << i << "  " << j << "  " << setprecision(20) << fPa[i*fNwave + j] << endl; 
  // 	}
  //   } 
  //cout << setprecision(20) << nll << endl;

  return -2.0*nll;
} 

void OpenPWA::randomizeParameters(int seed)
{
  TRandom3 r(seed);
  for(int i = 0; i < fNpar; i++)
    {
      if(fParInfo[i].step == 0)
	{
	  continue;
	}

      //fParVal[i] = fParVal[i] + r.Rndm()*(fParInfo[i].max - fParInfo[i].min);
      fParVal[i] = -200. + r.Rndm()*400.;
      fParInfo[i].step = fabs(fParVal[i]/10.);
    }
} 

void OpenPWA::printParameters()
{
  for(int i = 0; i < fNpar; i++)
    {
      cout << fParInfo[i].index << "  " << fParVal[i] << "  " << fParInfo[i].step << "  "
           << fParInfo[i].tag << "  " << fParInfo[i].name << "  " << fParInfo[i].min << "  "
           << fParInfo[i].max << endl; 
    }
}

void OpenPWA::setAllParValue(const double *par)
{
  for(int i = 0; i < fNpar; i++)
    {
      fParVal[i] = cl_float(par[i]);
    }
}

void OpenPWA::getAllParValue(double *par)
{
  for(int i = 0; i < fNpar; i++)
    {
      par[i] = double(fParVal[i]);
    }
}

void OpenPWA::convertAmplitudeBES()
{
  for(int i = 0; i < fNwave; i++)
    {
      double phase = fParVal[2*i+1]*PI/180.;
      fAmp[i] = double(fParVal[2*i])*TComplex(TMath::Cos(phase), TMath::Sin(phase));
    }

  fAmp[2] = fAmp[1]*double(fParVal[4]);
  fAmp[3] = fAmp[1]*double(fParVal[6]);

  //need to set the phase of the partial waves of one resonance to same

  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  TComplex inter = fAmp[i]*TComplex::Conjugate(fAmp[j]);
	  if(i == j) fPa[i*fNwave+j] = inter.Rho();
	  if(i < j) fPa[i*fNwave+j] = 2.*inter.Re();
	  if(i > j) fPa[i*fNwave+j] = 2.*inter.Im();
	}
    }
}

void OpenPWA::convertAmplitudeBES_grad()
{
  for(int i = 0; i < fNwave; i++)
    {
      double phase = fParVal[2*i+1]*PI/180.;
      fAmp[i] = double(fParVal[2*i])*TComplex(TMath::Cos(phase), TMath::Sin(phase));
    }

  fAmp[2] = fAmp[1]*double(fParVal[4]);
  fAmp[3] = fAmp[1]*double(fParVal[6]);

  //need to set the phase of the partial waves of one resonance to same

  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  TComplex inter = fAmp[i]*TComplex::Conjugate(fAmp[j]);
	  if(i == j) fPa_grad[i*fNwave+j] = inter.Rho();
	  if(i < j) fPa_grad[i*fNwave+j] = 2.*inter.Re();
	  if(i > j) fPa_grad[i*fNwave+j] = 2.*inter.Im();
	}
    }
}

void OpenPWA::convertAmplitudeFDC()
{
  for(int i = 0; i < fNpar/2; i++)
    {
      fAmp_FDC[i] = TComplex(fParVal[2*i], fParVal[2*i+1]);
    }

  fAmp[0] = fAmp_FDC[0]*fAmp_FDC[2];
  fAmp[1] = fAmp_FDC[1]*fAmp_FDC[2];
  fAmp[3] = fAmp_FDC[3]*fAmp_FDC[5];
  fAmp[4] = fAmp_FDC[4]*fAmp_FDC[5];

  //need to set the phase of the partial waves of one resonance to same

  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  TComplex inter = fAmp[i]*TComplex::Conjugate(fAmp[j]);
	  if(i == j) fPa[i*fNwave+j] = inter.Rho();
	  if(i < j) fPa[i*fNwave+j] = inter.Re();
	  if(i > j) fPa[i*fNwave+j] = inter.Im();
	}
    }
}

void OpenPWA::convertAmplitudeFDC_grad()
{
  for(int i = 0; i < fNpar/2; i++)
    {
      fAmp_FDC[i] = TComplex(fParVal[2*i], fParVal[2*i+1]);
    }

  fAmp[0] = fAmp_FDC[0]*fAmp_FDC[2];
  fAmp[1] = fAmp_FDC[1]*fAmp_FDC[2];
  fAmp[3] = fAmp_FDC[3]*fAmp_FDC[5];
  fAmp[4] = fAmp_FDC[4]*fAmp_FDC[5];
  //need to set the phase of the partial waves of one resonance to same

  for(int i = 0; i < fNwave; i++)
    {
      for(int j = 0; j < fNwave; j++)
	{
	  TComplex inter = fAmp[i]*TComplex::Conjugate(fAmp[j]);
	  if(i == j) fPa_grad[i*fNwave+j] = inter.Rho();
	  if(i < j) fPa_grad[i*fNwave+j] = inter.Re();
	  if(i > j) fPa_grad[i*fNwave+j] = inter.Im();
	}
    }
}


void OpenPWA::saveOptResult(ROOT::Math::Minimizer *minimizer)
{
  FitResult result;

  result.status = minimizer->Status();
  result.s = minimizer->MinValue();

  const double *val = minimizer->X();
  const double *err = minimizer->Errors();

  for(int i = 0; i < fNpar; i++)
    {
      result.val[i] = val[i];
      result.err[i] = err[i];
    } 

  fFitResults.push_back(result);
}

void OpenPWA::showBestResult()
{
  sort(fFitResults.begin(), fFitResults.end(), resultEval);

  cout << "Fit status: " << fFitResults[0].status << endl;
  cout << "S = 2lnL: " << fFitResults[0].s << endl;

  for(int i = 0; i < fNpar; i++)
    {
      cout << i << "  " << fFitResults[0].val[i] << "  " << fFitResults[0].err[i] << endl;
    }
}

void OpenPWA::showTopResults(unsigned int n)
{
  sort(fFitResults.begin(), fFitResults.end(), resultEval);

  if(n > fFitResults.size())
    {
      n = fFitResults.size();
    }

  cout << endl;
  if(fEnable_CL)
    {
      cout << "Results are based on GPU" << endl << endl;
    }
  else
    {
      cout << "Results are based on CPU" << endl << endl;
    }

  for(unsigned int i = 0; i < n; i++)
    {
      cout << "========== The top " << i << " result =============" << endl;
      cout << "Fit status: " << fFitResults[i].status << endl;
      cout << "S = 2lnL: " << setprecision(15) << fFitResults[i].s << endl;
    }
}

string OpenPWA::readKernelFile(string filename)
{
  size_t size;
  char *str;

  fstream fin(filename.c_str(), (std::fstream::in | std::fstream::binary));

  if(fin.is_open())
    {
      size_t sizeFile;

      //Find the size of the stream
      fin.seekg(0, std::fstream::end);
      size = sizeFile = fin.tellg();
      fin.seekg(0, std::fstream::beg);

      str = new char[size + 1];

      fin.read(str, sizeFile);
      fin.close();
      str[size] = '\0';

      string content = str;
      delete[] str;

      return content;
    }
  else
    {
      cerr << "Kernel file doesn't exist !! " << endl;
      exit(0);
    }
}

void OpenPWA::initializeCL()
{
  cl_int status[50];
  int idx = 0;
  for(int i = 0; i < 50; i++)
    {
      status[i] = -999;
    }

  fTime = (double *)malloc(sizeof(double)*5);
  for(int i = 0; i < 5; i++) 
    {
      fTime[i] = 0.;
    }

  //WorkItem, WorkGroup configuration
  globalSize = fNbunch/nPrefold;
  nBlocks = globalSize/(cl_int)localSize;
  
  //Get available devices, and pick AMD device
  cl_uint nPlatforms;
  status[idx++] = clGetPlatformIDs(0, NULL, &nPlatforms);
  if(nPlatforms < 0)
    {
      cerr << "OpenCL device not found!! Check the SDK and catalyst driver!" << endl;
      exit(0);
    }

  cl_platform_id *platforms = new cl_platform_id[nPlatforms];
  status[idx++] = clGetPlatformIDs(nPlatforms, platforms, NULL);

  for(unsigned i = 0; i < nPlatforms; i++)
    {
      char buffer[100];
      status[idx++] = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);

      platform = platforms[i];
      if(!strcmp(buffer, "Advanced Micro Devices, Inc."))
  	{
  	  break;
  	}
    }
  delete[] platforms;

  //Create context with AMD platform
  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
  context = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status[idx++]);

  //Get device list and choose the 1st device for calculation
  size_t nDevice;
  status[idx++] = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &nDevice);

  devices = (cl_device_id *)malloc(nDevice);
  status[idx++] = clGetContextInfo(context, CL_CONTEXT_DEVICES, nDevice, devices, NULL);

  queue = clCreateCommandQueue(context, devices[0], 0, &status[idx++]);

  //Decide the workgroup size, etc.
  size_t maxWorkgroupSize = 0;
  status[idx++] = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void *)&maxWorkgroupSize, NULL);

  cl_uint maxWorkItemDim = 0;
  status[idx++] = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), (void *)&maxWorkItemDim, NULL);

  size_t *maxWorkItemSizes = (size_t *)malloc(maxWorkItemDim*sizeof(size_t));
  status[idx++] = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*maxWorkItemDim, (void *)maxWorkItemSizes, NULL);

  logDebug(cout << "Info: " << maxWorkgroupSize << "  " << maxWorkItemDim << "  " << maxWorkItemSizes[0] << endl);

  //Create IO buffers
  fNll = (cl_float *)malloc(nBlocks*(1+fNwave2)*sizeof(cl_float));
  //fGrad = (cl_float *)malloc(nBlocks*fNwave2*sizeof(cl_float));
  for(int i = 0; i < nBlocks; i++)
    {
      fNll[i] = 0.;
      for(int j = 0; j < fNwave; j++)
	{
	  for(int k = 0; k < fNwave; k++)
	    {
	      //fGrad[i*fNwave2+j*fNwave+k] = 0.;
	      fNll[nBlocks+i*fNwave2+j*fNwave+k] = 0.;
	    }
	}
    }
  
  buffer_funall = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float4)*fNbunch*fNwave2, fFunall, &status[idx++]);
  buffer_nll = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float)*nBlocks*(1+fNwave2), fNll, &status[idx++]);
  //buffer_grad = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float)*nBlocks*fNwave2, fGrad, &status[idx++]);
  buffer_par = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float)*fNwave2, fPa, &status[idx++]);

  //Create and build program  
  string temp = readKernelFile("Kernel.cl");
  const char *source = temp.c_str();
  size_t length[] = { strlen(source) };

  logDebug(cout << "Content of the kernel file: " << endl);
  logDebug(cout << temp << endl);

  program = clCreateProgramWithSource(context, 1, &source, length, &status[idx++]);
  status[idx++] = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

  kernel = clCreateKernel(program, "dcs", &status[idx++]);

  size_t kernelWorkgroupSize = 0;
  status[idx++] = clGetKernelWorkGroupInfo(kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkgroupSize, 0);

  logDebug(cout << "Info: " << kernelWorkgroupSize << endl);

  //Set the invraiant argument
  status[idx++] = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer_funall);
  status[idx++] = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffer_nll);
  status[idx++] = clSetKernelArg(kernel, 2, localSize*sizeof(cl_float4), NULL);
  status[idx++] = clSetKernelArg(kernel, 3, localSize*fNwave2*sizeof(cl_float4), NULL);
  status[idx++] = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&buffer_par);
  status[idx++] = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&fNwave);
  status[idx++] = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&fNwave2);
  status[idx++] = clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&fNbunch);

  status[idx++] = clEnqueueWriteBuffer(queue, buffer_par, CL_TRUE, 0, fNwave2*sizeof(cl_float), fPa, 0, NULL, NULL);
  status[idx++] = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event_kernel);
  status[idx++] = clEnqueueReadBuffer(queue, buffer_nll, CL_FALSE, 0, nBlocks*(1+fNwave2)*sizeof(cl_float), fNll, 1, &event_kernel, &event_nll);
  //status[idx++] = clEnqueueReadBuffer(queue, buffer_grad, CL_FALSE, 0, nBlocks*fNwave2*sizeof(cl_float), fGrad, 1, &event_kernel, &event_grad);

  status[idx++] = clWaitForEvents(1, &event_nll);
  //status[idx++] = clWaitForEvents(1, &event_grad);

  for(int i = 0; i < idx; i++)
    {
      if(status[i] != CL_SUCCESS)
	{
	  cout << "OpenCL initialization failed at step " << i << " with error code: " << status[i] << endl;
	  finalizeCL();
	  exit(0);
	}
    }

  return;
}

void OpenPWA::finalizeCL()
{
  cl_int status[50];
  int idx = 0;
  for(int i = 0; i < 50; i++)
    {
      status[i] = 999;
    }

  for(int i = 0; i < 5; i++) cout << i << "  " << setprecision(20) << fTime[i] << endl;

  status[idx++] = clReleaseKernel(kernel);  
  status[idx++] = clReleaseProgram(program);  

  status[idx++] = clReleaseMemObject(buffer_funall);  
  status[idx++] = clReleaseMemObject(buffer_nll);  
  //status[idx++] = clReleaseMemObject(buffer_grad);
  status[idx++] = clReleaseMemObject(buffer_par); 

  status[idx++] = clReleaseCommandQueue(queue);  
  status[idx++] = clReleaseContext(context); 

  free(devices);  
  free(fNll);  
  free(fTime);
  //free(fGrad);

}


