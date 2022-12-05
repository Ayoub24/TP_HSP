#include <time.h>
#include <stdio.h>

void MatrixInit(float *M, int n, int p) {
    for(int i = 0; i < p; i++) {
        for(int j = 0; j < n; j++) {
            M[i*n+j] = float(rand())/(float(RAND_MAX)/2.0)-1.0;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    printf("[");
    for (int row=0; row<n; row++)
    {
        for(int col=0; col<p; col++)
        {
            if (row==n-1 & col==p-1){
                printf("%f]", M[row*p+col]);
            }else{
                printf("%f    ", M[row*p+col]);
            }
        }
        printf("\n");
    }
}

double MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    clock_t t;
    t = clock();
    for(int i = 0; i < p; i++) {
        for(int j = 0; j < n; j++) {
            Mout[i*n+j] = M1[i*n+j] + M2[i*n+j];
        }
    }
    t = clock() - t;
    printf("%ld\n\n", t);
    return double(t)/double(CLOCKS_PER_SEC);
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    clock_t t;
    t = clock();
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while(tid < n*p) {
        Mout[tid] = M1[tid] + M2[tid];
        tid += blockDim.x;
    }
    t = clock() - t;
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    int sum = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            sum = 0;
            for(int k = 0; k < n; k++) {
                sum += M1[i*n+k]*M2[k*n+j];
            }
            Mout[i*n+j] = sum;
        }
    }
}

int main() {
    double CPUtime = 0;
    double *GPUtime = 0;
    int n = 5;
    int p = 5;
    float *M1;
    float *M2;
    float *M3;
    float *M4;
    M1 = (float *) malloc(sizeof(float)*p*n);
    M2 = (float *) malloc(sizeof(float)*p*n);
    M3 = (float *) malloc(sizeof(float)*p*n);
    M4 = (float *) malloc(sizeof(float)*p*n);
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    CPUtime = MatrixAdd(M1, M2, M3, n, p);
    MatrixPrint(M1, n, p);
    printf("\n");
    MatrixPrint(M2, n, p);
    printf("\n");
    MatrixPrint(M3, n, p);
    printf("\n");

    float *cM1;
    float *cM2;
    float *cM3;
    if(cudaMalloc((float**)&cM1, sizeof(float)*p*n) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed.");
        exit(1);
    }
    if(cudaMalloc((float**)&cM2, sizeof(float)*p*n) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed.");
        exit(1);
    }
    if(cudaMalloc((float**)&cM3, sizeof(float)*p*n) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed.");
        exit(1);
    }
    if(cudaMemcpy(cM1, M1, sizeof(float)*p*n, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed.");
        exit(1);
    }
    if(cudaMemcpy(cM2, M2, sizeof(float)*p*n, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed.");
        exit(1);
    }

    cudaMatrixAdd<<<n,p>>>(cM1, cM2, cM3, n, p);
    if(cudaMemcpy(M4, cM3, sizeof(float)*p*n, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed.");
        exit(1);
    }

    MatrixPrint(M4, n, p);

    printf("\n\nCPU time : %f\n", CPUtime);

    free(M1);
    free(M2);
    free(M3);
    free(M4);
    cudaFree(cM1);
    cudaFree(cM2);
    cudaFree(cM3);

    return 0;
}

