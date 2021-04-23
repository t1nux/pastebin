#include "hip/hip_runtime.h"
#include <thrust/complex.h>
#include <eigen3/Eigen/Core>
#include "hipfft.h"
#include <fstream>
#include <iostream>
#include <stack>
#include <ctime>

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}

hipfftHandle fftPlan;
hipfftResult rc;

Eigen::ArrayXcd z;
Eigen::ArrayXcd Z;
Eigen::ArrayXcd z_test;
thrust::complex<double> *z_d;

void fft(thrust::complex<double>* z_d) {
    rc = hipfftExecZ2Z(fftPlan,
                       reinterpret_cast<hipfftDoubleComplex *>(z_d),
                       reinterpret_cast<hipfftDoubleComplex *>(z_d),
                       HIPFFT_FORWARD);
    assert(rc == HIPFFT_SUCCESS);
    hipDeviceSynchronize();
}

void ifft(thrust::complex<double>* z_d) {
    rc = hipfftExecZ2Z(fftPlan,
                       reinterpret_cast<hipfftDoubleComplex *>(z_d),
                       reinterpret_cast<hipfftDoubleComplex *>(z_d),
                       HIPFFT_BACKWARD);
    assert(rc == HIPFFT_SUCCESS);
    hipDeviceSynchronize();
}

void export_data(Eigen::ArrayXcd z, std::string fn, int N) {
    std::ofstream fl;
    fl.open(fn, std::ios::out | std::ios::binary);
    for (int i=0; i < N; ++i) {
        double rp = z[i].real();
        double ip = z[i].imag();
        fl.write(reinterpret_cast<char*>(&rp), sizeof(double));
        fl.write(reinterpret_cast<char*>(&ip), sizeof(double));
    }
    fl.close();
}

int main() {
    int Nt = 1<<12;
    int Ny = 1<<7;
    int Nx = 1<<6;

    z = Eigen::ArrayXcd::Zero(Nt*Nx*Ny);
    Z = Eigen::ArrayXcd::Zero(Nt*Nx*Ny);
    z_test = Eigen::ArrayXcd::Zero(Nt*Nx*Ny);
    hipMalloc(reinterpret_cast<void **>(&z_d), sizeof(hipDoubleComplex)*Nt*Nx*Ny);
    hipDeviceSynchronize();

    rc = HIPFFT_SUCCESS;
    fftPlan = NULL;
    rc = hipfftCreate(&fftPlan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan3d(&fftPlan, Nt, Ny, Nx, HIPFFT_Z2Z);
    assert(rc == HIPFFT_SUCCESS);
    hipDeviceSynchronize();

    // set some to 1
    int fac = 8;
    for (int tt=0; tt<Nt/fac; ++tt) {
        for (int yy=0; yy<Ny/fac; ++yy) {
            for (int xx=0; xx<Nx/fac; ++xx) {
                z[Ny*Nx*tt + Nx*yy + xx] = 1.;
            }
        }
    }

    hipMemcpy(z_d, z.data(), Nt*Ny*Nx*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    tic();

    for (int i=0; i<100; ++i) {
        fft(z_d);
        hipDeviceSynchronize();
        ifft(z_d);
        hipDeviceSynchronize();
    }


    toc();

    std::cout << Nt << std::endl;
    return 0;
}
