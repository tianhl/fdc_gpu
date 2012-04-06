
  float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
  for(int i = 0; i < nWave; i++)
    {
      for(int j = 0; j < nWave; j++)
        {
          int idx = i*nWave + j;
          sum += pPa[i*nWave+idx]*pFunall[gid*nWave2+idx];
        }
    }

  tempNll[lid] = log(sum);
  for(int i = 0; i < nWave; i++)
    {
      for(int j = 0; j < nWave; j++)
        {
          int idx = i*nWave + j;
          tempGrad[lid*nWave2+idx] = sum*pFunall[gid*nWave2+idx];
        }
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
      pRes[bid] = tempRes[0].x;  
      pRes[bid] += tempRes[0].y; 
      pRes[bid] += tempRes[0].z; 
      pRes[bid] += tempRes[0].w;

      for(int i = 0; i < nWave; i++)
        {  
          for(int j = 0; j < nWave; j++)
            {
              int idx = i*nWave + j;
              pRes[nBlocks+bid*nWave2+idx] = tempGrad[idx].x;
              pRes[nBlocks+bid*nWave2+idx] += tempGrad[idx].y;
              pRes[nBlocks+bid*nWave2+idx] += tempGrad[idx].z;
              pRes[nBlocks+bid*nWave2+idx] += tempGrad[idx].w;
            }
        }        
    }
