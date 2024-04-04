#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pdbio.h"
#include "xyzio.h"
#include "dcdio.h"
#include "md.h"
#include "ran2.h"
#include "hicio.h"
#include "ht.cu"

int rseed = -86775414;
int seed = 742425;

#define BUF_SIZE 256

#define kB 8.314462e-3	   // Boltzman constant (kJ/K/mol)
#define gamma 50.0

//**************************************************************new types

typedef struct {
	int typei;
	int typej;
	float A;
	float B;
} LJPar;

typedef struct {
	int strCount;
	LJPar* pars;
} LJData;

//**************************************************************

int N = 5000;
float L = 19.0f;
float sigma= 1.0;
float epsilon = 1.0;
float T = 300.0f;
int T0 = 2;

//**************************************************************

float dt = 0.001f;
long int totalsteps = 100000000;
int stride = 10000;

//**************************************************************

MDSystem mds;
DCD dcd;

//**************************************************************

#define l 2

float A[l][l];
float B[l][l];
float2* h_AB;
float2* d_AB;

//**************************************************************

//int* h_pairsCount;
int* d_pairsCount;
//int* h_pairs;
int* d_pairs;
float cutoff = 2.0;
int pairs_freq = 50;

//**************************************************************

inline __host__ __device__ float len(float3 a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

inline __host__ __device__ float3 getVector(float3 ri, float3 rj, float L){
	float3 dr;	
	dr.x = rj.x - ri.x;
	dr.y = rj.y - ri.y;
	dr.z = rj.z - ri.z;
	dr.x -= rint(dr.x/L)*L;
	dr.y -= rint(dr.y/L)*L;
	dr.z -= rint(dr.z/L)*L;
	return dr;
}

inline __host__ __device__ float getDistance(float3 ri, float3 rj, float L){
	float3 dr = getVector(ri, rj, L);
	return len(dr);
}

/*inline __host__ __device__ float3 transferPBC(float3 r, float L){
	if(r.x > L){
		r.x -= L;
	} else
	if(r.x < 0){
		r.x += L;
	}
	if(r.y > L){
		r.y -= L;
	} else
	if(r.y < 0){
		r.y += L;
	}
	if(r.z > L){
		r.z -= L;
	} else
	if(r.z < 0){
		r.z += L;
	}
	return r;
}*/ //Transfer [0,L]


// Transfer [-L,L]

inline __host__ __device__ float3 transferPBC(float3 r, float L){
	r.x -= rint(r.x/L)*L;
	r.y -= rint(r.y/L)*L;
	r.z -= rint(r.z/L)*L;
	return r;
}

//**************************************************************

void readLJPar(const char* filename, LJData* ljData){
	printf("Reading %s.\n", filename);
	char buffer[BUF_SIZE];
	FILE* file = fopen(filename, "r");
	if(file != NULL){
		fgets(buffer, BUF_SIZE, file);
		ljData->strCount = 4;
		ljData->pars = (LJPar*)calloc(ljData->strCount, sizeof(LJPar));
		int i;
		char* pch;
		for(i = 0; i < ljData->strCount; i++){
			fgets(buffer, BUF_SIZE, file);
			pch = strtok(buffer, " \t\r\n");
			ljData->pars[i].typei = atoi(pch);
			pch = strtok(NULL, " \t\r\n");
			ljData->pars[i].typej = atoi(pch);
			pch = strtok(NULL, " \t\r\n");
			ljData->pars[i].A = atof(pch);
			pch = strtok(NULL, " \t\r\n");
			ljData->pars[i].B = atof(pch);
		}
	printf("Done reading '%s'.\n", filename);
	fclose(file);
	} else {
		perror(filename);
		exit(0);
	}

	int p;	
	h_AB = (float2*)calloc(T0*T0,sizeof(float2));	
	cudaMalloc((void**)&d_AB, T0*T0*sizeof(float2));	
	for (p = 0; p < ljData->strCount; p++){
		int typei = ljData->pars[p].typei;
		int typej = ljData->pars[p].typej;

		h_AB[typej + T0*typei].x = ljData->pars[p].A;
		h_AB[typej + T0*typei].y = ljData->pars[p].B;
	}
	cudaMemcpy(d_AB, h_AB, T0*T0*sizeof(float2), cudaMemcpyHostToDevice);
	int i;
	int j;
	printf("A:\n");
	for(i = 0; i < l; i++){
		for(j = 0; j < l; j++){
			printf("%f\t%f\t", h_AB[j + T0*i].x, h_AB[j + T0*i].y);
		}
		printf("\n");
	}
		
}

//**************************************************************

__global__ void computeLJ(float3* r, float3* f, int* type, float sigma, float epsilon, int N, float L, float2* d_AB, int T0){
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 fi = f[i];
		float3 ri = r[i];
		for(j = 0; j < N; j++){
			if (j != i){
				float3 rj = r[j];
				int typei = type[i];
				int typej = type[j];
				float2 AB = d_AB[typej + T0*typei];
				/*float Aij = A[typei][typej];
				float Bij = B[typei][typej];*/
//				float Aij = 1.0f;
//				float Bij = -2.0f;
				float3 rij = getVector(ri, rj, L);
				float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
				float srij2 = sigma*sigma/rijmod2;	
				float srij6 = srij2*srij2*srij2;
				float df = -6.0f*epsilon*srij6*(2.0f*AB.x*srij6 + AB.y)/rijmod2;;

				fi.x += df*rij.x;
				fi.y += df*rij.y;
				fi.z += df*rij.z;
			}
		}
		f[i] = fi;
	}
}

//**************************************************************

__global__ void computeLJPairlist(float3* r, float3* f, int* type, float sigma, float epsilon, int N, float L, float2* d_AB, int T0, int* d_pairsCount, int* d_pairs){
	int i, j, p;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 fi = f[i];
		float3 ri = r[i];
		for(p = 0; p < d_pairsCount[i]; p++){
			j = d_pairs[p*N + i];
			float3 rj = r[j];
			int typei = type[i];
			int typej = type[j];
			float2 AB = d_AB[typej + T0*typei];
			float3 rij = getVector(ri, rj, L);
			float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
			float srij2 = sigma*sigma/rijmod2;	
			float srij6 = srij2*srij2*srij2;
			float df = -6.0f*epsilon*srij6*(2.0f*AB.x*srij6 + AB.y)/rijmod2;;

			fi.x += df*rij.x;
			fi.y += df*rij.y;
			fi.z += df*rij.z;
		}
		f[i] = fi;
	}
}

//**************************************************************

void Velocities(){
	int i;
	for(i = 0; i < N; i++){
		double mult = sqrt(kB*T/mds.h_m[i]);
		mds.h_v[i].x = mult*gasdev(&rseed);
		mds.h_v[i].y = mult*gasdev(&rseed);
		mds.h_v[i].z = mult*gasdev(&rseed);
	}
}

//**************************************************************

__global__ void integrateGPU(float3* d_v, float3* d_r, float3* d_f, float* d_m, float dt, int N, double var, float L){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){

		float4 rf = rforce(i);

		float3 f = d_f[i];
		float3 r = d_r[i];
		float3 v = d_v[i];
		float m = d_m[i];

		f.x += -gamma*v.x + var*rf.x;
		f.y += -gamma*v.y + var*rf.y;
		f.z += -gamma*v.z + var*rf.z;		

		v.x += dt*f.x/m;
		v.y += dt*f.y/m;
		v.z += dt*f.z/m;

		r.x += v.x*dt;
		r.y += v.y*dt;
		r.z += v.z*dt;

		f.x = 0.0;
		f.y = 0.0;
		f.z = 0.0;

		r = transferPBC(r, L);

		d_r[i] = r;
		d_v[i] = v;	
		d_f[i] = f;
	}

}

//**************************************************************

__global__ void PairlistGPU(float3* r, int N, float L, int* d_pairsCount, int* d_pairs, float cutoff){
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 ri = r[i];
		int pairsCount = 0;
		for(j = 0; j < N; j++){
			if (j != i){
				float3 rj = r[j];
				if(getDistance(ri, rj, L) < cutoff) {
					d_pairs[pairsCount*N + i] = j;
					pairsCount ++;
				}
			}
		}
		d_pairsCount[i] = pairsCount;
	}				
}


//**************************************************************

int main(int argc, char* argv[]){

	cudaSetDevice(1);

// Init molecular system

	XYZ xyz;
	readXYZ("Atomscoor.xyz", &xyz);
		
	mds.N = xyz.atomCount;
	mds.h_type = (int*)calloc(mds.N, sizeof(int));
	
	mds.h_r = (float3*)calloc(mds.N, sizeof(float3));
	mds.h_v = (float3*)calloc(mds.N, sizeof(float3));
	mds.h_f = (float3*)calloc(mds.N, sizeof(float3));
	mds.h_m = (float*)calloc(mds.N, sizeof(float));

	cudaMalloc((void**)&mds.d_type, mds.N*sizeof(float3));	
	cudaMalloc((void**)&mds.d_r, mds.N*sizeof(float3));
	cudaMalloc((void**)&mds.d_v, mds.N*sizeof(float3));
	cudaMalloc((void**)&mds.d_f, mds.N*sizeof(float3));
	cudaMalloc((void**)&mds.d_m, mds.N*sizeof(float));
	
	createDCD(&dcd, mds.N, 11, 0, 1.0, 1, 0, 0, 0, 0);
	dcdOpenWrite(&dcd, "dynam.dcd");
	dcdWriteHeader(dcd);

	int i;
	for(i = 0; i < mds.N; i++){
		mds.h_type[i] = atoi(&xyz.atoms[i].name);
		mds.h_r[i].x = xyz.atoms[i].x/10.0;
		mds.h_r[i].y = xyz.atoms[i].y/10.0;
		mds.h_r[i].z = xyz.atoms[i].z/10.0;
		mds.h_m[i] = 1.0;
	}

	Velocities();

	cudaMemcpy(mds.d_type, mds.h_type, mds.N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_r, mds.h_r, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_v, mds.h_v, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_f, mds.h_f, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_m, mds.h_m, mds.N*sizeof(float), cudaMemcpyHostToDevice);

// Init LJ data
	
	LJData lj;
	readLJPar("LJPar", &lj);

// Init pairlist
	
	cudaMalloc((void**)&d_pairsCount, mds.N*sizeof(int));
	cudaMalloc((void**)&d_pairs, 1024*mds.N*sizeof(int));

// Langevin integrator
	
	initRand(seed, mds.N);

	double var = sqrtf(kB*T*2.0*gamma/dt);
	

// Main cycle	

	int blockSize = 512;
	int blockNum = mds.N/blockSize + 1;

	long int step;
	for(step = 0; step < totalsteps; step++){
		if(step % pairs_freq == 0){
			PairlistGPU<<<blockNum, blockSize>>>(mds.d_r, mds.N, L, d_pairsCount, d_pairs, cutoff);
		}

		computeLJPairlist<<<blockNum, blockSize>>>(mds.d_r, mds.d_f, mds.d_type, sigma, epsilon, mds.N, L, d_AB, T0, d_pairsCount, d_pairs);
		integrateGPU<<<blockNum, blockSize>>>(mds.d_v, mds.d_r, mds.d_f, mds.d_m, dt, mds.N, var, L);
		int i;
		if(step % stride == 0){
			cudaMemcpy(mds.h_r, mds.d_r, mds.N*sizeof(float3), cudaMemcpyDeviceToHost);
			for(i = 0; i < mds.N; i++){
				dcd.frame.X[i] = 10.0*mds.h_r[i].x;
				dcd.frame.Y[i] = 10.0*mds.h_r[i].y;
				dcd.frame.Z[i] = 10.0*mds.h_r[i].z;
			}
			dcdWriteFrame(dcd);
			printf("Step %ld\n", step);
		}
	}
	return 1;
}
