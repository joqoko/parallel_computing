// include c++ header files
#include <iostream>
#include "cuda_runtime.h"

// include local CUDA header files.
#include "include/cuda_kernel.cuh"

using namespace std;

int main(int argc, char** argv)
{
  cout << "test" << endl;
  cudaError_t cudaStatus;

  int num = 0;
  cudaStatus = cudaGetDeviceCount(&num);
  cout << "Number of GPU [" << num << "]" << endl;

  cudaDeviceProp prop;
  if(num > 0)
  {
    cudaGetDeviceProperties(&prop, 0);
    cout << "Device [" << prop.name << "]" << endl;
  }


  return 0;

}
