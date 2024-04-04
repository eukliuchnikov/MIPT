#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pdbio.h"
#include "psfio.h"
#include "xyzio.h"
#include "dcdio.h"
#include "md.h"
#include "ran2.h"
//#include "hicio.h"
#include "ht.cu"

#include <curand.h>
#include <curand_kernel.h>

int rseed = -436462;
int seed = 346964;

#define BUF_SIZE 256

#define kB 8.314462e-3	   // Boltzman constant (kJ/K/mol) 
#define gamma 22600  // g/mol/ps

#define kconst 1.0

/*#define typeA 0
#define typeB 1
#define typeC 2*/
#define typeCount 10
int typeCounts[typeCount];

//**************************************************************

typedef struct {
	int typei;
	int typej;
	float A;
	float B;
	int e;
	float sigma;
} LJPar;

typedef struct {
	int strCount;
	LJPar* pars;
} LJData;

//**************************************************************

//int N = 9290;	//The number of particles
float3 L = make_float3(270.0f, 270.0f, 270.0f); 	// Size of box, nm
//float sigma = 2.0;	// Min range between particles, nm
//float epsilon = 5.0;	// Just coef
float T = 300.0f;	//Temperature, K
//D(diffusion coef) [nm2/ps]

//**************************************************************

float dt = 5.0f;	//ps
long int totalsteps = 10000000;
int stride = 1000;
int dcdstride = 10000;

//**************************************************************

MDSystem mds;
DCD dcd;
float3* start;

//**************************************************************

float A[typeCount][typeCount];
float B[typeCount][typeCount];
float4* h_AB;
float4* d_AB;

//**************************************************************

//int* h_pairsCount;
int* d_pairsCount;
//int* h_pairs;
int* d_pairs;

int* d_pairs2Count;
int* d_pairs2;
float cutoff = 16.0;
int pairs_freq = 10;

float cutoff2 = 40.0;
int pairs2_freq = 1000;

//**************************************************************

inline __host__ __device__ float len(float3 a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

inline __host__ __device__ float3 getVector(float3 ri, float3 rj, float3 L){
	float3 dr;	
	dr.x = rj.x - ri.x;
	dr.y = rj.y - ri.y;
	dr.z = rj.z - ri.z;
	dr.x -= rint(dr.x/L.x)*L.x;
	dr.y -= rint(dr.y/L.y)*L.y;
	dr.z -= rint(dr.z/L.z)*L.z;
	return dr;
}

inline __host__ __device__ float getDistance(float3 ri, float3 rj, float3 L){
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

inline __host__ __device__ float3 transferPBC(float3 r, float3 L){
	r.x -= rint(r.x/L.x)*L.x;
	r.y -= rint(r.y/L.y)*L.y;
	r.z -= rint(r.z/L.z)*L.z;
	return r;
}

//**************************************************************

void readLJPar(const char* filename, LJData* ljData){
	printf("Reading %s.\n", filename);
	char buffer[BUF_SIZE];
	FILE* file = fopen(filename, "r");
	if(file != NULL){
		fgets(buffer, BUF_SIZE, file);
		ljData->strCount = typeCount*typeCount;
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
			pch = strtok(NULL, " \t\r\n");
			ljData->pars[i].e = atoi(pch);
			pch = strtok(NULL, " \t\r\n");
			ljData->pars[i].sigma = atof(pch);
		}
	printf("Done reading '%s'.\n", filename);
	fclose(file);
	} else {
		perror(filename);
		exit(0);
	}

	int p;	
	h_AB = (float4*)calloc(typeCount*typeCount,sizeof(float4));	
	cudaMalloc((void**)&d_AB, typeCount*typeCount*sizeof(float4));	

	for (p = 0; p < ljData->strCount; p++){
		int typei = ljData->pars[p].typei;
		int typej = ljData->pars[p].typej;

		h_AB[typej + typeCount*typei].x = ljData->pars[p].A;
		h_AB[typej + typeCount*typei].y = ljData->pars[p].B;
		h_AB[typej + typeCount*typei].z = ljData->pars[p].e;
		h_AB[typej + typeCount*typei].w = ljData->pars[p].sigma;
		
	}
	cudaMemcpy(d_AB, h_AB, typeCount*typeCount*sizeof(float4), cudaMemcpyHostToDevice);

	int i;
	int j;
	printf("A:\n");
	for(i = 0; i < typeCount; i++){
		for(j = 0; j < typeCount; j++){
			printf("(%f; %f)_(%f; %f)\t", h_AB[j + typeCount*i].x, h_AB[j + typeCount*i].y, h_AB[j + typeCount*i].z, h_AB[j + typeCount*i].w);
		}
		printf("\n");

	}
		
}

//**************************************************************

__global__ void computeLJ(float3* r, float3* f, int* type, float sigma, float epsilon, int N, float3 L, float2* d_AB){
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
				float2 AB = d_AB[typej + typeCount*typei];
				/*float Aij = A[typei][typej];
				float Bij = B[typei][typej];*/
				//float Aij = 1.0f;
				//float Bij = -2.0f;
				float3 rij = getVector(ri, rj, L);
				float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
				float srij2 = sigma*sigma/rijmod2;	
				float srij6 = srij2*srij2*srij2;
				float df = -6.0f*epsilon*srij6*(2.0f*AB.x*srij6 + AB.y)/rijmod2;
//				float df = -6.0f*epsilon*srij6*(2.0f*Aij*srij6 + Bij)/rijmod2;

				fi.x += df*rij.x;
				fi.y += df*rij.y;
				fi.z += df*rij.z;
			}
		}
		f[i] = fi;
	}
}

//**************************************************************

__global__ void computeLJPairlist(float3* r, float3* f, int* type, int N, float3 L, float4* d_AB, int* d_pairsCount, int* d_pairs){
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
			float4 AB = d_AB[typej + typeCount*typei];
//			float Aij = 0.0f;
//			float Bij = 1.0f;
			float3 rij = getVector(ri, rj, L);
			float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
			float srij2 = AB.w*AB.w/rijmod2;	
//			float srij6 = srij2*srij2*srij2;
			float df = -2.0f*srij2*AB.z*(2.0f*AB.x*srij2 + AB.y)/rijmod2;
//			float df = -6.0f*epsilon*srij6*(2.0f*AB.x*srij6 + AB.y)/rijmod2;
//			float df = -6.0f*epsilon*srij6*(2.0f*Aij*srij6 + Bij)/rijmod2;

			fi.x += df*rij.x;
			fi.y += df*rij.y;
			fi.z += df*rij.z;
		}
		f[i] = fi;
	}
}

//**************************************************************

void velocities(int N){
	int i;
	for(i = 0; i < N; i++){
		double mult = sqrt(kB*T/mds.h_m[i]);
		mds.h_v[i].x = mult*gasdev(&rseed);
		mds.h_v[i].y = mult*gasdev(&rseed);
		mds.h_v[i].z = mult*gasdev(&rseed);
	}
}

//**************************************************************

__global__ void integrateGPU(float3* d_v, float3* d_r, float3* d_f, float* d_m, int* type, float dt, int N, double var, float3 L){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){

		float4 rf = rforce(i);

		float3 f = d_f[i];
		float3 r = d_r[i];
//		float3 v = d_v[i];
//		float m = d_m[i];

		r = transferPBC(r, L);
		
/*		f.x += -gamma*v.x + var*rf.x;
		f.y += -gamma*v.y + var*rf.y;
		f.z += -gamma*v.z + var*rf.z;		

		v.x += dt*f.x/m;
		v.y += dt*f.y/m;
		v.z += dt*f.z/m;

		r.x += v.x*dt;
		r.y += v.y*dt;
		r.z += v.z*dt;
*/
		f.x += var*rf.x;
		f.y += var*rf.y;
		f.z += var*rf.z;

		r.x += f.x*dt/gamma;
		r.y += f.y*dt/gamma;
		r.z += f.z*dt/gamma;

		f.x = 0.0f;
		f.y = 0.0f;
		f.z = 0.0f;

		d_r[i] = r;
//		d_v[i] = v;	
		d_f[i] = f;
	}

}

//**************************************************************

__global__ void pairlistGPU(float3* r, int N, float3 L, int* d_pairsCount, int* d_pairs, int* d_pairs2Count, int* d_pairs2, float cutoff){
	int i, j, p;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 ri = r[i];
		int pairsCount = 0;
		for(p = 0; p < d_pairs2Count[i]; p++){
			j = d_pairs2[p*N + i];			
			float3 rj = r[j];
			if(getDistance(ri, rj, L) < cutoff) {
				d_pairs[pairsCount*N + i] = j;
				pairsCount ++;
			}
		}
		d_pairsCount[i] = pairsCount;
	}				
}

//**************************************************************

__global__ void pairlist2GPU(float3* r, int N, float3 L, int* d_pairs2Count, int* d_pairs2, float cutoff2){
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 ri = r[i];
		int pairsCount = 0;
		for(j = 0; j < N; j++){
			if (j != i){
				float3 rj = r[j];
				if(getDistance(ri, rj, L) < cutoff2) {
					d_pairs2[pairsCount*N + i] = j;
					pairsCount ++;
				}
			}
		}
		d_pairs2Count[i] = pairsCount;
	}				
}


//**************************************************************

__global__ void interaction(float3* r, int* type, float4* d_AB, int* d_pairsCount, int* d_pairs, int N, float3 L, float dt, int typeA, int typeB, int typeC, float pr, curandState* devState, double o, double s){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){
		int typei = type[i];
		if(typei == typeA){
			float3 ri = r[i];
			for(int p = 0; p < d_pairsCount[i]; p++){
				int j = d_pairs[p*N + i];
				int typej = type[j];
				float4 AB = d_AB[typej + typeCount*typei];
				if(typej == typeB){
					float3 rj = r[j];
//					float c = randht(i);
					double c = curand_uniform_double(&devState[i]);
					double u = curand_uniform_double(&devState[i]);
					double a = curand_uniform_double(&devState[i]);
					float rij = getDistance(ri, rj, L);
					//if(rij < 1.0f*sigma && c < 0.0000000014f){
					if(rij < (4.0f*AB.w +1.0f) && c < pr/(rij*rij*rij) && u < o && a < s){		
						typei = typeC;
					}
				}
			}
			type[i] = typei;
		}
	}
}


//**************************************************************

void countTypes(){
	int i;
	for(i = 0; i < typeCount; i++){
		typeCounts[i] = 0;
	}
	cudaMemcpy(mds.h_type, mds.d_type, mds.N*sizeof(int), cudaMemcpyDeviceToHost);
	for(i = 0; i < mds.N; i++){
		int typei = mds.h_type[i];
		typeCounts[typei]++; 
	}	
}

//**************************************************************

void checkCUDAError(){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("CUDA error: %s \n", cudaGetErrorString(error));
exit(0);
	}
}

//**************************************************************

__global__ void initCurand(curandState *state, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, idx, &state[idx]);
}


//**************************************************************

int main(int argc, char* argv[]){

	cudaSetDevice(0);

// Init molecular system
	
	int type11 = 0;
	int type11a = 5;
	int type9 = 1;
	int type9a = 6;
	int type10 = 2;
	int type10a = 7;
	int type2 = 3;
	int type2a = 8;
	int type1 = 4;
	int type1a = 9;

	double k11 = 1.0e-6;
	double k9 = 1.0e-3;
	double k10 = 1.0e-6;
	double k2 = 1.0;
	double k1 = 1.0e-2;

	double kapp = 1.0e-2;
	double kabs = 1.0;

	float alpha = 2.28 + 0.13897*log(kconst);
	float pr = alpha*kconst*dt/(4.0f*M_PI/3.0f);
//	float pr1 = pr/((4.0f*sigma +1.0f)*(4.0f*sigma +1.0f)*(4.0f*sigma +1.0f));
//	printf("%e\n", pr1);
//	float pr = kconst*dt/(4.0f*M_PI/3.0f);

	XYZ xyz;
	readXYZ("Atomscoor.xyz", &xyz);
	
	mds.N = xyz.atomCount;
	start = (float3*)calloc(mds.N, sizeof(float3));	
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
	
	createDCD(&dcd, 10*mds.N, 11, 0, 1.0, 1, 0, 0, 0, 0);
	dcdOpenWrite(&dcd, "dynam_rect_cas_big.dcd");
	dcdWriteHeader(dcd);

	int i;

	PSF psf;
	psf.natom = mds.N*10;
	psf.atoms = (PSFAtom*)calloc(psf.natom, sizeof(PSFAtom));
	psf.nbond = 0;
	psf.ntheta = 0;
	psf.nphi = 0;
	psf.nimphi = 0;
	psf.ncmap = 0;
	psf.nnb = 0;
	for(i = 0; i < psf.natom; i++){
		psf.atoms[i].id = i + 1;
		if(i < mds.N){
			sprintf(psf.atoms[i].name, "11");
			sprintf(psf.atoms[i].type, "11");
			sprintf(psf.atoms[i].segment, "11");
		} else if(i < 2*mds.N){
			sprintf(psf.atoms[i].name, "11a");
			sprintf(psf.atoms[i].type, "11a");
			sprintf(psf.atoms[i].segment, "11a");
		} else if(i < 3*mds.N){
			sprintf(psf.atoms[i].name, "9");
			sprintf(psf.atoms[i].type, "9");
			sprintf(psf.atoms[i].segment, "9");
		} else if(i < 4*mds.N){
			sprintf(psf.atoms[i].name, "9a");
			sprintf(psf.atoms[i].type, "9a");
			sprintf(psf.atoms[i].segment, "9a");
		} else if(i < 5*mds.N){
			sprintf(psf.atoms[i].name, "10");
			sprintf(psf.atoms[i].type, "10");
			sprintf(psf.atoms[i].segment, "10");
		} else if(i < 6*mds.N){
			sprintf(psf.atoms[i].name, "10a");
			sprintf(psf.atoms[i].type, "10a");
			sprintf(psf.atoms[i].segment, "10a");
		} else if(i < 7*mds.N){
			sprintf(psf.atoms[i].name, "2");
			sprintf(psf.atoms[i].type, "2");
			sprintf(psf.atoms[i].segment, "2");
		} else if(i < 8*mds.N){
			sprintf(psf.atoms[i].name, "2a");
			sprintf(psf.atoms[i].type, "2a");
			sprintf(psf.atoms[i].segment, "2a");
		} else if(i < 9*mds.N){
			sprintf(psf.atoms[i].name, "1");
			sprintf(psf.atoms[i].type, "1");
			sprintf(psf.atoms[i].segment, "1");
		} else {
			sprintf(psf.atoms[i].name, "1a");
			sprintf(psf.atoms[i].type, "1a");
			sprintf(psf.atoms[i].segment, "1a");
		}
		sprintf(psf.atoms[i].resName, "CEL");
		psf.atoms[i].resid = i + 1;
		psf.atoms[i].m = 1.0;
		psf.atoms[i].q = 0.0;

	}

	char filename[1024];

	sprintf(filename, "fibrin_rect_cas_big.psf");
	writePSF(filename, &psf);

	for(i = 0; i < mds.N; i++){
		mds.h_type[i] = atoi(&xyz.atoms[i].name);
		mds.h_r[i].x = xyz.atoms[i].x/10.0;
		mds.h_r[i].y = xyz.atoms[i].y/10.0;
		mds.h_r[i].z = xyz.atoms[i].z/10.0;
		mds.h_r[i] = transferPBC(mds.h_r[i], L);
		mds.h_m[i] = 36000.0;
	}
//	for(i = 0; i < 5000; i++){
//		mds.h_type[i] = 2;
//	}
	
	//velocities(mds.N);

	cudaMemcpy(mds.d_type, mds.h_type, mds.N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_r, mds.h_r, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_v, mds.h_v, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_f, mds.h_f, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_m, mds.h_m, mds.N*sizeof(float), cudaMemcpyHostToDevice);

// Init LJ data
	
	LJData lj;
//	readLJPar("LJPar_cas_MM", &lj);
	readLJPar("LJPar_cas_MM", &lj);

// Init pairlist
	
	cudaMalloc((void**)&d_pairsCount, mds.N*sizeof(int));
	cudaMalloc((void**)&d_pairs, 1500*mds.N*sizeof(int));

	cudaMalloc((void**)&d_pairs2Count, mds.N*sizeof(int));
	cudaMalloc((void**)&d_pairs2, 1500*mds.N*sizeof(int));

// Langevin integrator
	int blockSize = 512;
	int blockNum = mds.N/blockSize + 1;
	
	curandState *devState;
	cudaMalloc((void**)&devState, mds.N * sizeof(curandState));
 	initCurand<<<blockNum, blockSize>>>(devState, seed);
	cudaDeviceSynchronize();
	
	initRand(seed, mds.N);

	double var = sqrtf(kB*T*2.0*gamma/dt);
	

// Main cycle	

	long int step;
	FILE * fp;

   	fp = fopen ("fibrin_test_cas_big.dat", "w");

	for(i = 0; i < mds.N; i++){
		start[i].x = mds.h_r[i].x;
		start[i].y = mds.h_r[i].y;
		start[i].z = mds.h_r[i].z;
	}
	printf("1\n");
	checkCUDAError();

	printf("Starting simulations\n");
	for(step = 0; step <= totalsteps; step++){
		int i;
		double average;	
		double average1;
		average = 0.0;
		checkCUDAError();
		if(step % stride == 0){
			cudaMemcpy(mds.h_r, mds.d_r, mds.N*sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(mds.h_type, mds.d_type, mds.N*sizeof(int), cudaMemcpyDeviceToHost);
			int type11ID = 0;
			int type11aID = mds.N;
			int type9ID = 2*mds.N;
			int type9aID = 3*mds.N;
			int type10ID = 4*mds.N;
			int type10aID = 5*mds.N;
			int type2ID = 6*mds.N;
			int type2aID = 7*mds.N;
			int type1ID = 8*mds.N;
			int type1aID = 9*mds.N;
			for(i = 0; i < mds.N; i++){
	
				double dx = mds.h_r[i].x - start[i].x;
				double dy = mds.h_r[i].y - start[i].y;	
				double dz = mds.h_r[i].z - start[i].z;		
				
				average1 = dx*dx + dy*dy + dz*dz; 
				average += average1;

				mds.h_r[i] = transferPBC(mds.h_r[i], L);
				
				if(mds.h_type[i] == 0){		
					dcd.frame.X[type11ID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type11ID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type11ID] = 10.0*mds.h_r[i].z;
					type11ID ++;
				} 
				else if(mds.h_type[i] == 5){
					dcd.frame.X[type11aID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type11aID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type11aID] = 10.0*mds.h_r[i].z;
					type11aID ++;
				} 
				else if(mds.h_type[i] == 1){
					dcd.frame.X[type9ID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type9ID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type9ID] = 10.0*mds.h_r[i].z;
					type9ID ++;
				}
				else if(mds.h_type[i] == 6){		
					dcd.frame.X[type9aID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type9aID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type9aID] = 10.0*mds.h_r[i].z;
					type9aID ++;
				} 
				else if(mds.h_type[i] == 2){		
					dcd.frame.X[type10ID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type10ID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type10ID] = 10.0*mds.h_r[i].z;
					type10ID ++;
				} 
				else if(mds.h_type[i] == 7){		
					dcd.frame.X[type10aID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type10aID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type10aID] = 10.0*mds.h_r[i].z;
					type10aID ++;
				} 
				else if(mds.h_type[i] == 3){
					dcd.frame.X[type2ID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type2ID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type2ID] = 10.0*mds.h_r[i].z;
					type2ID ++;
				} 
				else if(mds.h_type[i] == 8){
					dcd.frame.X[type2aID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type2aID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type2aID] = 10.0*mds.h_r[i].z;
					type2aID ++;
				}
				else if(mds.h_type[i] == 4){		
					dcd.frame.X[type1ID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type1ID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type1ID] = 10.0*mds.h_r[i].z;
					type1ID ++;
				} 
				else if(mds.h_type[i] == 9){		
					dcd.frame.X[type1aID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[type1aID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[type1aID] = 10.0*mds.h_r[i].z;
					type1aID ++;
				} 
			}

			//cudaMemcpy(mds.d_r, mds.h_r, mds.N*sizeof(float3), cudaMemcpyHostToDevice);

			for(i = type11ID; i < mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type11aID; i < 2*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type9ID; i < 3*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type9aID; i < 4*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type10ID; i < 5*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type10aID; i < 6*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type2ID; i < 7*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type2aID; i < 8*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type1ID; i < 9*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = type1aID; i < 10*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}

			average = 100.0*average/mds.N;
			float stept = step*dt;
			countTypes();
			fprintf(fp, "%lf\t%f\t",stept, average);
//			printf("%lf\t%f\t",stept, average);
			for(i = 0; i < typeCount; i++){
				fprintf(fp, "%d\t",typeCounts[i]);
//				printf("%d\t",typeCounts[i]);
			}
			fprintf(fp, "\n");
//			printf("\n");

			if(step % dcdstride == 0){
				dcdWriteFrame(dcd);
			}
//			printf("Step %ld\n", step);
		}
		if(step % pairs2_freq == 0){
			pairlist2GPU<<<blockNum, blockSize>>>(mds.d_r, mds.N, L, d_pairs2Count, d_pairs2, cutoff2);
		}
		if(step % pairs_freq == 0){
			pairlistGPU<<<blockNum, blockSize>>>(mds.d_r, mds.N, L, d_pairsCount, d_pairs, d_pairs2Count, d_pairs2, cutoff);
		}

		computeLJPairlist<<<blockNum, blockSize>>>(mds.d_r, mds.d_f, mds.d_type, mds.N, L, d_AB, d_pairsCount, d_pairs);
		
		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, d_AB, d_pairsCount, d_pairs, mds.N, L, dt, type11, type2a, type11a, pr, devState, k11, kapp);
		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, d_AB, d_pairsCount, d_pairs, mds.N, L, dt, type9, type11a, type9a, pr, devState, k9, kabs);
		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, d_AB, d_pairsCount, d_pairs, mds.N, L, dt, type10, type9a, type10a, pr, devState, k10, kabs);
		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, d_AB, d_pairsCount, d_pairs, mds.N, L, dt, type2, type10a, type2a, pr, devState, k2, kabs);
		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, d_AB, d_pairsCount, d_pairs, mds.N, L, dt, type1, type2a, type1a, pr, devState, k1, kabs);



		integrateGPU<<<blockNum, blockSize>>>(mds.d_v, mds.d_r, mds.d_f, mds.d_m, mds.d_type, dt, mds.N, var, L);	
	}
	fclose(fp);
	return 1;
}
