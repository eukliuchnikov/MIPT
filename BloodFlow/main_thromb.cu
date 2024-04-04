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
#include "hicio.h"
#include "ht.cu"

int rseed = -9354472;
int seed = 673954;

#define BUF_SIZE 256

#define kB 8.314462e-3	   // Boltzman constant (kJ/K/mol) 
#define gamma 22600  // g/mol/ps

//#define alpha 0.25
#define kconst 1.0

/*#define typeA 0
#define typeB 1
#define typeC 2*/
#define typeCount 3
int typeCounts[typeCount];

//**************************************************************

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

//int N = 5000;	//The number of particles
float3 L = make_float3(1000.0f, 100.0f, 100.0f); 	// Size of box, nm
float sigma = 2.0;	// Min range between particles, nm
float epsilon = 200.0;	// Just coef
float T = 300.0f;	//Temperature, K
//D(diffusion coef) [nm2/ps]

//**************************************************************

float dt = 0.1f;	//ps
long int totalsteps = 30000000;
int stride = 1000;
int dcdstride = 1000;

//**************************************************************

MDSystem mds;
DCD dcd;
float3* start;

//**************************************************************

float A[typeCount][typeCount];
float B[typeCount][typeCount];
float2* h_AB;
float2* d_AB;

//**************************************************************

//int* h_pairsCount;
int* d_pairsCount;
//int* h_pairs;
int* d_pairs;
float cutoff = 10.0;
int pairs_freq = 1000;

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
	if(r.y < -0.5*L.y) {
		r.y = -L.y - r.y;
	} else if(r.y > 0.5*L.y){
		r.y = L.y - r.y;
	}	
	//r.y -= rint(r.y/L.y)*L.y;
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
		}
	printf("Done reading '%s'.\n", filename);
	fclose(file);
	} else {
		perror(filename);
		exit(0);
	}

	int p;	
	h_AB = (float2*)calloc(typeCount*typeCount,sizeof(float2));	
	cudaMalloc((void**)&d_AB, typeCount*typeCount*sizeof(float2));	
	for (p = 0; p < ljData->strCount; p++){
		int typei = ljData->pars[p].typei;
		int typej = ljData->pars[p].typej;

		h_AB[typej + typeCount*typei].x = ljData->pars[p].A;
		h_AB[typej + typeCount*typei].y = ljData->pars[p].B;
	}
	cudaMemcpy(d_AB, h_AB, typeCount*typeCount*sizeof(float2), cudaMemcpyHostToDevice);
	int i;
	int j;
	printf("A:\n");
	for(i = 0; i < typeCount; i++){
		for(j = 0; j < typeCount; j++){
			printf("(%f;%f)\t", h_AB[j + typeCount*i].x, h_AB[j + typeCount*i].y);
		}
		printf("\n");
	}
		
}

//**************************************************************

/*__global__ void computeLJ(float3* r, float3* f, int* type, float sigma, float epsilon, int N, float3 L, float2* d_AB, int typeCount){
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
/*				float3 rij = getVector(ri, rj, L);
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
}*/

//**************************************************************

__global__ void computeLJPairlist(float3* r, float3* f, int* type, float sigma, float epsilon, int N, float3 L, float2* d_AB, int* d_pairsCount, int* d_pairs){
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
			float2 AB = d_AB[typej + 3*typei];
//			float Aij = 0.0f;
//			float Bij = 1.0f;
			float3 rij = getVector(ri, rj, L);
			float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
			float srij2 = sigma*sigma/rijmod2;	
			float srij6 = srij2*srij2*srij2;
			float df = -6.0f*epsilon*srij6*(2.0f*AB.x*srij6 + AB.y)/rijmod2;;
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

		if(r.x > 0.5*L.x && type[i] == 2){
			type[i] = 0;
		}
		r = transferPBC(r, L);
		// Water drag
		f.x += 1.0*(r.y + 0.5*L.y);

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

		if(!(i >= 100 && i < 140)){
			r.x += f.x*dt/gamma;
			r.y += f.y*dt/gamma;
			r.z += f.z*dt/gamma;
		}

		f.x = 0.0f;
		f.y = 0.0f;
		f.z = 0.0f;

		d_r[i] = r;
//		d_v[i] = v;	
		d_f[i] = f;
	}

}

//**************************************************************

__global__ void pairlistGPU(float3* r, int N, float3 L, int* d_pairsCount, int* d_pairs, float cutoff){
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

__global__ void interaction(float3* r, int* type, float sigma, int* d_pairsCount, int* d_pairs, int N, float3 L, float dt, int typeA, int typeB, int typeC, float pr){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){
		int typei = type[i];
		if(typei == typeA){
			float3 ri = r[i];
			for(int p = 0; p < d_pairsCount[i]; p++){
				int j = d_pairs[p*N + i];
				int typej = type[j];
				if(typej == typeB){
					float3 rj = r[j];
					float c = randht(i);
					float dx = ri.x - rj.x;
					float dy = ri.y - rj.y;
					float dz = ri.z - rj.z;
					//float rij = getDistance(ri, rj, L);
					float rij = sqrtf(dx*dx + dy*dy + dz*dz);//getDistance(ri, rj, L);
					//if(rij < 1.0f*sigma && c < 0.0000000014f){
					if(c < pr/(rij*rij*rij)){		
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

int main(int argc, char* argv[]){

	cudaSetDevice(0);

// Init molecular system
	
	int type0 = 0;
	int type1 = 1;
	int type2 = 2;
//	int type3 = 3;
//	int type4 = 4;

	float pr = kconst*dt/(4.0f*M_PI/3.0f);

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
	
	createDCD(&dcd, 3*mds.N, 11, 0, 1.0, 1, 0, 0, 0, 0);
	dcdOpenWrite(&dcd, "dynam_thromb.dcd");
	dcdWriteHeader(dcd);

	int i;

	PSF psf;
	psf.natom = mds.N*3;
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
			sprintf(psf.atoms[i].name, "typeA");
			sprintf(psf.atoms[i].type, "0");
			sprintf(psf.atoms[i].segment, "0");
		} else if(i < 2*mds.N){
			sprintf(psf.atoms[i].name, "typeB");
			sprintf(psf.atoms[i].type, "1");
			sprintf(psf.atoms[i].segment, "1");
		} else {
			sprintf(psf.atoms[i].name, "typeC");
			sprintf(psf.atoms[i].type, "2");
			sprintf(psf.atoms[i].segment, "2");
		}
		sprintf(psf.atoms[i].resName, "CEL");
		psf.atoms[i].resid = i + 1;
		psf.atoms[i].m = 1.0;
		psf.atoms[i].q = 0.0;

	}

	char filename[1024];

	sprintf(filename, "fibrin_thromb.psf");
	writePSF(filename, &psf);

	for(i = 0; i < mds.N; i++){
		mds.h_type[i] = atoi(&xyz.atoms[i].name);
		mds.h_r[i].x = xyz.atoms[i].x/10.0;
		mds.h_r[i].y = xyz.atoms[i].y/10.0;
		mds.h_r[i].z = xyz.atoms[i].z/10.0;
		mds.h_r[i] = transferPBC(mds.h_r[i], L);
		mds.h_m[i] = 36000.0;
	}
	for(i = 0; i < 100; i++){
		//mds.h_r[i].x = -0.5*L.x + 30.0;
		//mds.h_r[i].y = -0.5*L.y;
		//mds.h_r[i].z = (i/20.0 - 0.5)*L.z;
		mds.h_type[i] = 1;
	}
	for(i = 100; i < 140; i++){
		mds.h_r[i].x = 0.0;
		mds.h_r[i].y = -0.5*L.y + 10.0;
		mds.h_r[i].z = ((i-20.0)/20.0 - 0.5)*L.z;
		mds.h_type[i] = 2;
	}

	//velocities(mds.N);

	cudaMemcpy(mds.d_type, mds.h_type, mds.N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_r, mds.h_r, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_v, mds.h_v, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_f, mds.h_f, mds.N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(mds.d_m, mds.h_m, mds.N*sizeof(float), cudaMemcpyHostToDevice);

// Init LJ data
	
	LJData lj;
	readLJPar("LJPar", &lj);

// Init pairlist
	
    /*h_pairsCount = (int*)calloc(mds.N, sizeof(int));
	h_pairs = (int*)calloc(1024*mds.N, sizeof(int));
    for(i = 0; i < mds.N; i++){
        h_pairsCount[i] = 0;
        for(int j = 0; j < 1024; j++){
            h_pairs[i*1024 + j] = 0;
        }
    }*/
	cudaMalloc((void**)&d_pairsCount, mds.N*sizeof(int));
	cudaMalloc((void**)&d_pairs, 1024*mds.N*sizeof(int));
    //cudaMemcpy(d_pairsCount, h_pairsCount, mds.N*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_pairs, h_pairs, 1024*mds.N*sizeof(int), cudaMemcpyHostToDevice);

// Langevin integrator
	
	initRand(seed, mds.N);

	double var = sqrtf(kB*T*2.0*gamma/dt);
	

// Main cycle	

	int blockSize = 512;
	int blockNum = mds.N/blockSize + 1;

	long int step;
	FILE * fp;

   	fp = fopen ("fibrin_thromb.dat", "w");

	for(i = 0; i < mds.N; i++){
		start[i].x = mds.h_r[i].x;
		start[i].y = mds.h_r[i].y;
		start[i].z = mds.h_r[i].z;
	}

	for(step = 0; step <= totalsteps; step++){
		int i;
		double average;	
		double average1;
		average = 0.0;

		if(step % stride == 0){
			cudaMemcpy(mds.h_r, mds.d_r, mds.N*sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(mds.h_type, mds.d_type, mds.N*sizeof(int), cudaMemcpyDeviceToHost);
			int typeAID = 0;
			int typeBID = mds.N;
			int typeCID = 2*mds.N;
			for(i = 0; i < mds.N; i++){
	
				double dx = mds.h_r[i].x - start[i].x;
				double dy = mds.h_r[i].y - start[i].y;	
				double dz = mds.h_r[i].z - start[i].z;		
				
				average1 = dx*dx + dy*dy + dz*dz; 
				average += average1;

				mds.h_r[i] = transferPBC(mds.h_r[i], L);
				
				if(mds.h_type[i] == 0){		
					dcd.frame.X[typeAID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[typeAID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[typeAID] = 10.0*mds.h_r[i].z;
					typeAID ++;
				} 
				else if(mds.h_type[i] == 1){
					dcd.frame.X[typeBID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[typeBID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[typeBID] = 10.0*mds.h_r[i].z;
					typeBID ++;
				} 
				else if(mds.h_type[i] == 2){
					dcd.frame.X[typeCID] = 10.0*mds.h_r[i].x;
					dcd.frame.Y[typeCID] = 10.0*mds.h_r[i].y;
					dcd.frame.Z[typeCID] = 10.0*mds.h_r[i].z;
					typeCID ++;
				}
			}

			//cudaMemcpy(mds.d_r, mds.h_r, mds.N*sizeof(float3), cudaMemcpyHostToDevice);

			for(i = typeAID; i < mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = typeBID; i < 2*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}
			for(i = typeCID; i < 3*mds.N; i++){
				dcd.frame.X[i] = 0.0;
				dcd.frame.Y[i] = 0.0;
				dcd.frame.Z[i] = 0.0;
			}

			average = 100.0*average/mds.N;
			float stept = step*dt;
			countTypes();
			fprintf(fp, "%lf\t%f\t",stept, average);
			printf("%lf\t%f\t",stept, average);
			for(i = 0; i < typeCount; i++){
				fprintf(fp, "%d\t",typeCounts[i]);
				printf("%d\t",typeCounts[i]);
			}
			fprintf(fp, "\n");
			printf("\n");

			if(step % dcdstride == 0){
				dcdWriteFrame(dcd);
			}
//			printf("Step %ld\n", step);
		}
		if(step % pairs_freq == 0){
			pairlistGPU<<<blockNum, blockSize>>>(mds.d_r, mds.N, L, d_pairsCount, d_pairs, cutoff);
		}

		computeLJPairlist<<<blockNum, blockSize>>>(mds.d_r, mds.d_f, mds.d_type, sigma, epsilon, mds.N, L, d_AB, d_pairsCount, d_pairs);
		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, sigma, d_pairsCount, d_pairs, mds.N, L, dt, type0, type1, type2, pr);
//		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, sigma, d_pairsCount, d_pairs, mds.N, L, dt, type2, type1, type3, pr);
//		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, sigma, d_pairsCount, d_pairs, mds.N, L, dt, type3, type1, type4, pr);
//		interaction<<<blockNum, blockSize>>>(mds.d_r,mds. d_type, sigma, d_pairsCount, d_pairs, mds.N, L, dt, type4, type1, type2, pr);



		integrateGPU<<<blockNum, blockSize>>>(mds.d_v, mds.d_r, mds.d_f, mds.d_m, mds.d_type, dt, mds.N, var, L);	
	}
	fclose(fp);
	return 1;
}
