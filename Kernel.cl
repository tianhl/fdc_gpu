__kernel
void 
dcs(const __global float4 *pFunall,
                  __global float *pRes,
                  __local float4 *tempNll,
                  __local float4 *tempGrad,
                  __constant float *pPa,
                  const int nWave,
                  const int nWave2,
                  const int nBunch)
{
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint bid = get_group_id(0);
  uint lsize = get_local_size(0);

  tempNll[lid] = (float4)(0.f, 0.f, 0.f, 0.f);
  for(int i = 0; i < nWave; i++)
    {
      for(int j = 0; j < nWave; j++)
        {
          int idx = i*nWave + j;
          tempGrad[lid*nWave2+idx] = (float4)(0.f, 0.f, 0.f, 0.f);
        }
    } 
 
  while(gid < nBunch)
    {
      float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
      for(int i = 0; i < nWave; i++)
        {
          for(int j = 0; j < nWave; j++)
            {
              int idx = i*nWave + j;
              sum += pPa[idx]*pFunall[gid*nWave2+idx];
            }
        }
 
      tempNll[lid] += log(sum);
      for(int i = 0; i < nWave; i++)
        {
          for(int j = 0; j < nWave; j++)
            {
              int idx = i*nWave + j;
              tempGrad[lid*nWave2+idx] += pFunall[gid*nWave2+idx]/sum;
            }
        }

      gid += get_global_size(0);
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = lsize/2; offset > 0; offset >>= 1)
    {
      if(lid < offset)
        {
          tempNll[lid] += tempNll[lid+offset];
          for(int i = 0; i < nWave; i++)
            {  
              for(int j = 0; j < nWave; j++)
                {
                  int idx = i*nWave + j;
                  tempGrad[lid*nWave2+idx] += tempGrad[(lid+offset)*nWave2+idx];
                }
            } 
        }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if(lid == 0)
    {
      int idx_initial = bid*(nWave2+1);
      pRes[idx_initial] = tempNll[0].x;  
      pRes[idx_initial] += tempNll[0].y; 
      pRes[idx_initial] += tempNll[0].z; 
      pRes[idx_initial] += tempNll[0].w; 
 
      for(int i = 0; i < nWave; i++)
        {  
          for(int j = 0; j < nWave; j++)
            {
              int idx_global = idx_initial + i*nWave + j + 1;
              int idx_local = i*nWave + j;

              pRes[idx_global] = tempGrad[idx_local].x;
              pRes[idx_global] += tempGrad[idx_local].y;
              pRes[idx_global] += tempGrad[idx_local].z;
              pRes[idx_global] += tempGrad[idx_local].w;
            }
        }  
    }
}

