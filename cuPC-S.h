/*
 * cuPC_S.h
 *
 *  Created on: Apr 16, 2019
 *      Author: behrooz
 */

#ifndef CUPC_S_H_
#define CUPC_S_H_
//===============================> Definition <===============================
#define bx  blockIdx.x
#define by  blockIdx.y
#define bz  blockIdx.z
#define tx  threadIdx.x
#define ty  threadIdx.y
#define tz  threadIdx.z
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX(x,y) ((x)>(y)?(x):(y))
//Note that EdgePerBlockLX * ParTestPerEdgeL1 < 1024

#define ParGivenL1 64
#define NumOfBlockForEachNodeL1 2
#define ParGivenL2 64
#define NumOfBlockForEachNodeL2 2
#define ParGivenL3 64
#define NumOfBlockForEachNodeL3 2
#define ParGivenL4 64
#define NumOfBlockForEachNodeL4 2
#define ParGivenL5 64
#define NumOfBlockForEachNodeL5 2
#define ParGivenL6 64
#define NumOfBlockForEachNodeL6 2
#define ParGivenL7 64
#define NumOfBlockForEachNodeL7 2
#define ParGivenL8 64
#define NumOfBlockForEachNodeL8 2
#define ParGivenL9 64
#define NumOfBlockForEachNodeL9 2
#define ParGivenL10 64
#define NumOfBlockForEachNodeL10 2
#define ParGivenL11 64
#define NumOfBlockForEachNodeL11 2
#define ParGivenL12 64
#define NumOfBlockForEachNodeL12 2
#define ParGivenL13 64
#define NumOfBlockForEachNodeL13 2
#define ParGivenL14 64
#define NumOfBlockForEachNodeL14 2
#define ML 14

//==========================> Function Declaration <==========================
__global__ void cal_Indepl0(double *C, int *G, double th, double *pMax, int n);
__global__ void cal_Indepl1(double *C, int *G, int *GPrime, int *mutex, int* Sepset, double* pMax, double th, int n);
__global__ void cal_Indepl3(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl4(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl5(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl2(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl6(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl7(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl8(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl9(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl10(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl11(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl12(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl13(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__global__ void cal_Indepl14(double *C, int *G, int* GPrime, int *mutex, int* Sepset, double* pMax, int n, double th);
__device__ void pseudoinversel2(double M2[][2], double M2Inv[][2]);
__device__ void pseudoinversel3(double M2[][3], double M2Inv[][3]);
__device__ void pseudoinversel4(double M2[][4], double M2Inv[][4], double v[][4], double *rv1, double *w, double res1[][4] );
__device__ void pseudoinversel5(double M2[][5], double M2Inv[][5], double v[][5], double *rv1, double *w, double res1[][5] );
__device__ void pseudoinversel6(double M2[][6], double M2Inv[][6], double v[][6], double *rv1, double *w, double res1[][6] );
__device__ void pseudoinversel7(double M2[][7], double M2Inv[][7], double v[][7], double *rv1, double *w, double res1[][7] );
__device__ void pseudoinversel8(double M2[][8], double M2Inv[][8], double v[][8], double *rv1, double *w, double res1[][8] );
__device__ void pseudoinversel9(double M2[][9], double M2Inv[][9], double v[][9], double *rv1, double *w, double res1[][9] );
__device__ void pseudoinversel10(double M2[][10], double M2Inv[][10], double v[][10], double *rv1, double *w, double res1[][10] );
__device__ void pseudoinversel11(double M2[][11], double M2Inv[][11], double v[][11], double *rv1, double *w, double res1[][11] );
__device__ void pseudoinversel12(double M2[][12], double M2Inv[][12], double v[][12], double *rv1, double *w, double res1[][12] );
__device__ void pseudoinversel13(double M2[][13], double M2Inv[][13], double v[][13], double *rv1, double *w, double res1[][13] );
__device__ void pseudoinversel14(double M2[][14], double M2Inv[][14], double v[][14], double *rv1, double *w, double res1[][14] );
extern "C" void Skeleton(double* C, int *P, int *G, double *Th, int *l, int *maxlevel, double *pMax, int* SepSet);

__global__ void Initialize (int *Mat, int n);
__global__ void scan_compact(int* G_Compact,  const int* G, const int n, int *nprime);
__global__ void SepSet_initialize(int *SepSet, int size);

__device__ void BINOM(int n, int k, int *out);
__device__ void IthCombination(int out[], int N, int P, int L);
__device__ double PYTHAG(double a, double b);
__device__ void inverse(double M2[][3], double M2Inv[][3]);


#endif /* CUPC_S_H_ */
