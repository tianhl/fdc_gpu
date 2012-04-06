#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#include <TStopwatch.h>

#include "OpenPWA.h"
#include "config.h"

using namespace std;

OpenPWA *pwa;

int main(int argc, char *argv[])
{
  int iCase = atoi(argv[1]);

  pwa = new OpenPWA("pwa.config");
  pwa->init();

  if(iCase == 0)
    {
      cout << "Test NLL ... " << endl;
      cout << setprecision(20) << pwa->nll_cpu() << "  " << setprecision(20) << pwa->nll_cl() << endl;
      cout << setprecision(20) << pwa->nll_cpu() << "  " << setprecision(20) << pwa->nll_cl() << endl;
    }

  if(iCase == 1)
    {
      cout << "Test Gradients ... " << endl;

      double val1 = pwa->nll_cpu();
      double *grad = pwa->getParGradients();
      
      cout << setprecision(20) << endl;
      for(int i = 0; i < pwa->getNpar(); i++)
	{
	  cout << "CPU " << i << "  " << grad[i] << endl;
	}
      
      cout << setprecision(20) << val1 << endl;
      
      double val2 = pwa->nll_cl();
      for(int i = 0; i < pwa->getNpar(); i++)
	{
	  cout << "GPU " << i << "  " << grad[i] << endl;
	}
      cout << setprecision(20) << val2 << endl;
    }
  
  if(iCase == 2)
    {
      cout << "Test OpenCL time ... " << endl;

      double val1;
      TStopwatch timer;
      timer.Start();
      for(int i = 0; i < 10000; i++)
	{
	  val1 = pwa->nll_cl();
	}
      timer.Stop();

      cout << timer.CpuTime() << endl;
    }

  if(iCase == 3)
    {
      cout << "Test CPU time ... " << endl;

      double val1;
      TStopwatch timer;
      timer.Start();
      for(int i = 0; i < 10000; i++)
	{
	  val1 = pwa->nll_cpu();
	}
      timer.Stop();

      cout << timer.CpuTime() << endl;
    }

  delete pwa;
  return 1;
}
