#include <iostream>
#include <vector>

#include <TMinuit.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TF1.h>
#include <TAxis.h>
#include <TLine.h>
#include <TStopwatch.h>
#include <TFile.h>
#include <TTree.h>
#include <TMath.h>

#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "Math/Functor.h"
#include "Math/Factory.h"

#include "OpenPWA.h"
#include "config.h"

using namespace std;

OpenPWA *pwa;

int main(int argc, char *argv[])
{
  pwa = new OpenPWA("pwa.config");
  pwa->init();

  // cout << pwa->nll_cpu()/2. << endl;

  // return 0;

  double par[pwa->getNpar()];
  pwa->getAllParValue(par);

  Config fitConfig("pwa.config");

  ROOT::Math::GradFunctor fcn;
  if(fitConfig.pBool("Enable_CL"))
    {
      fcn = ROOT::Math::GradFunctor(pwa, &OpenPWA::Eval_GPU, &OpenPWA::Derivative, pwa->getNpar());
    }
  else
    {
      fcn = ROOT::Math::GradFunctor(pwa, &OpenPWA::Eval_CPU, &OpenPWA::Derivative, pwa->getNpar());
    }

  ROOT::Math::Minimizer *minimizer = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
  minimizer->SetMaxFunctionCalls(fitConfig.pDouble("MaxFunctionCalls"));
  minimizer->SetMaxIterations(fitConfig.pDouble("MaxIterations"));
  minimizer->SetTolerance(fitConfig.pDouble("Tolerance"));
  minimizer->SetFunction(fcn);
  minimizer->SetPrintLevel(fitConfig.pInt("PrintLevel"));

  //cout << nll(par) << "===" << pwa->nll_cpu() << endl;
  pwa->printParameters();


  int nIter = fitConfig.pInt("nIteration");
  int seed = fitConfig.pInt("seed");
  TStopwatch timer;
  timer.Start();
  for(int i = 0; i < nIter; i++)
    {
      pwa->randomizeParameters(seed + i);
      pwa->getAllParValue(par);

      //cout << i << "CPU: " << par[0] << "  " << pwa->getParValue(0) << "  " << pwa->nll_cpu() << endl;
      //cout << i << "GPU: " << par[0] << "  " << pwa->getParValue(0) << "  " << pwa->nll_cl() << endl;

      for(int j = 0; j < pwa->getNpar(); j++)
      	{
      	  ParameterInfo temp = pwa->getParInfo(j);
      	  minimizer->SetLimitedVariable(j, (temp.name).c_str(), pwa->getParValue(j), temp.step, temp.min, temp.max);
      	}
      minimizer->Minimize();

      pwa->saveOptResult(minimizer);
    }
  timer.Stop();

  pwa->showTopResults(10);
  pwa->showBestResult();
  // pwa->setParValue(par);

  cout << "The time consumption is: " << timer.CpuTime() << endl; 

  delete pwa;
  return 1;
}
