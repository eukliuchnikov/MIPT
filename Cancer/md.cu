#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dcdio.h"
#include "ran2.h"
#include "ht.cu"
#include "psfio.h"

int rseed = -789798;
int seed = 32456;

#define T 300 //K
#define Kb 8.314462e-3 //In kJ/K*mol
#define tau 0.001
#define gamma 50.0
#define maxN 5000000

#define maxDrivers 9 
#define pu 0.00034
#define pv 0.0001
#define ps 0.3
#define pc 0.001
#define psigma 0.1
#define pres 0.001


int N;

int* h_N;
float3* h_r;
float3* h_v;
float3* h_f;
float* h_m;
int* h_type;
int* h_drivers;

int* d_N;
float3* d_r;
float3* d_v;
float3* d_f;
float* d_m;
int* d_type;
int* d_drivers;


int* d_pairsCount;
int* d_pairs;

int* h_pairs2Count;

int* d_pairs2Count;
int* d_pairs2;
float cutoff = 0.25;
int pairs_freq = 1;

float cutoff2 = 0.5;
int pairs2_freq = 10;


DCD dcd;




inline __host__ __device__ float len(float3 a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

inline __host__ __device__ float3 getVector(float3 ri, float3 rj){
	float3 dr;	
	dr.x = rj.x - ri.x;
	dr.y = rj.y - ri.y;
	dr.z = rj.z - ri.z;
	return dr;
}

inline __host__ __device__ float getDistance(float3 ri, float3 rj){
	float3 dr = getVector(ri, rj);
	return len(dr);
}


__global__ void repulsion_kernel(float3* d_r, float3* d_f, int* d_type, int N);
__global__ void integrate_kernel(float3* d_v, float3* d_r, float3* d_f, float* d_m, int* d_type, int N, float var);
__global__ void division_death_kernel(float3* d_r, int* d_type, int* d_drivers, int* d_N, int N, int* d_pairsCount);
//__global__ void pairlistGPU(float3* r, int* d_pairsCount, int* d_pairs, int* d_pairs2Count, int* d_pairs2, float cutoff, int N);
__global__ void pairlist2GPU(float3* r, int* d_pairs2Count, int* d_pairs2, float cutoff2, int N);

void saveCoord(long int step);

int main(int argc, char* argv[]){

	int i, j, k;	

	cudaSetDevice(3);
	
	N = 10;
	h_N = (int*)calloc(1, sizeof(int));
	h_r = (float3*)calloc(maxN, sizeof(float3));
	h_v = (float3*)calloc(maxN, sizeof(float3));
	h_f = (float3*)calloc(maxN, sizeof(float3));
	h_m = (float*)calloc(maxN, sizeof(float));
	h_type = (int*)calloc(maxN, sizeof(int));
	h_drivers = (int*)calloc(maxN, sizeof(int));
	h_pairs2Count = (int*)calloc(maxN, sizeof(int));

	cudaMalloc((void**)&d_N, 1*sizeof(int));
	cudaMalloc((void**)&d_r, maxN*sizeof(float3));
	cudaMalloc((void**)&d_v, maxN*sizeof(float3));
	cudaMalloc((void**)&d_f, maxN*sizeof(float3));
	cudaMalloc((void**)&d_m, maxN*sizeof(float));
	cudaMalloc((void**)&d_type, maxN*sizeof(int));
	cudaMalloc((void**)&d_drivers, maxN*sizeof(int));
//	cudaMalloc((void**)&d_pairsCount, maxN*sizeof(int));
//	cudaMalloc((void**)&d_pairs, 1500*maxN*sizeof(int));
	cudaMalloc((void**)&d_pairs2Count, maxN*sizeof(int));
//	cudaMalloc((void**)&d_pairs2, 150*maxN*sizeof(int));

	h_N[0] = N;
	for(i = 0; i < maxN; i++){
		h_r[i].x = 0;
		h_r[i].y = 0;
		h_r[i].z = 0;
		h_type[i] = 0;
		h_m[i] = 1.0;
		float mult = sqrtf(Kb*T/h_m[i]);
		h_v[i].x = mult*gasdev(&rseed);
		h_v[i].y = mult*gasdev(&rseed);
		h_v[i].z = mult*gasdev(&rseed);
	}
	for(i = 1; i < N; i++){
		h_r[i].x = 1.0*gasdev(&rseed);
		h_r[i].y = 1.0*gasdev(&rseed);
		h_r[i].z = 1.0*gasdev(&rseed);
		float p = ran2(&rseed);
		if(p < 0.0){
			h_type[i] = 2;
		} else {
			h_type[i] = 1;
		}
	}

	cudaMemcpy(d_N, h_N, 1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, maxN*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, h_v, maxN*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, h_f, maxN*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, h_m, maxN*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_type, h_type, maxN*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_drivers, h_drivers, maxN*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairs2Count, h_pairs2Count, maxN*sizeof(int), cudaMemcpyHostToDevice);

	initRand(seed, maxN);


	PSF psf;
	psf.natom = maxN*3;
	psf.atoms = (PSFAtom*)calloc(psf.natom, sizeof(PSFAtom));
	psf.nbond = 0;
	psf.ntheta = 0;
	psf.nphi = 0;
	psf.nimphi = 0;
	psf.ncmap = 0;
	psf.nnb = 0;
	for(i = 0; i < psf.natom; i++){
		psf.atoms[i].id = i + 1;
		if(i < maxN){
			sprintf(psf.atoms[i].name, "N");
			sprintf(psf.atoms[i].type, "N");
			sprintf(psf.atoms[i].segment, "N");
		} else if(i < 2*maxN){
			sprintf(psf.atoms[i].name, "R");
			sprintf(psf.atoms[i].type, "R");
			sprintf(psf.atoms[i].segment, "R");
		} else {
			sprintf(psf.atoms[i].name, "D");
			sprintf(psf.atoms[i].type, "D");
			sprintf(psf.atoms[i].segment, "D");
		}
		sprintf(psf.atoms[i].resName, "CEL");
		psf.atoms[i].resid = i + 1;
		psf.atoms[i].m = 1.0;
		psf.atoms[i].q = 0.0;

	}

	char filename[1024];

	sprintf(filename, "output/colony.psf");
	writePSF(filename, &psf);

	
	createDCD(&dcd, maxN*3, 11, 0, 1.0, 1, 0, 0, 0, 0);
	sprintf(filename, "output/colony.dcd");
	dcdOpenWrite(&dcd, filename);
	dcdWriteHeader(dcd);	

	float var = sqrtf(Kb*T*2.0*gamma/tau);

	long int step;

	int blockSize = 512;
	int blockNum = N/blockSize + 1;

	FILE* out = fopen("output/colony.dat", "w");
	for(step = 0; step < 20000; step++){
		//printf("Step: %ld\n", step);
		if(step % 10 == 0){
			saveCoord(step);
		}

		blockNum = N/blockSize + 1;

		if(step % pairs2_freq == 0){
			pairlist2GPU<<<blockNum, blockSize>>>(d_r, d_pairs2Count, d_pairs2, cutoff2, N);
		}
/*		if(step % pairs_freq == 0){
			pairlistGPU<<<blockNum, blockSize>>>(d_r, d_pairsCount, d_pairs, d_pairs2Count, d_pairs2, cutoff);
		}*/
		//repulsion_kernel<<<blockNum, blockSize>>>(d_r, d_f, d_type, N);
		integrate_kernel<<<blockNum, blockSize>>>(d_v, d_r, d_f, d_m, d_type, N, var);

		if(step % 10 == 0){
			division_death_kernel<<<blockNum, blockSize>>>(d_r, d_type, d_drivers, d_N, N, d_pairs2Count);
			cudaMemcpy(h_N, d_N, 1*sizeof(int), cudaMemcpyDeviceToHost);
			N = h_N[0];
			//printf("Cell count: %d\n", N);
			if(N >= maxN || N == 0){
				exit(0);
			}
		}
		if(step % 10 == 0){
			int countRes = 0;
			int countNotres = 0;
			int countAlive = 0;
			int countDead = 0;
			int countDrivers = 0;
			int neighbors = 0;
			cudaMemcpy(h_type, d_type, maxN*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_drivers, d_drivers, maxN*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_pairs2Count, d_pairs2Count, maxN*sizeof(int), cudaMemcpyDeviceToHost);
			for(j = 0; j < N; j++){
				if(h_type[j] == 1){
					countNotres ++;
				}
				if(h_type[j] == 2){
					countRes ++;
				}
				if(h_type[j] == 3){
					countDead ++;
				}			
				if(h_type[j] > 0 && h_type[j] != 3){
					countAlive ++;
					countDrivers += h_drivers[j];
				}
			}

			for(k = 0; k < maxN; k++){
				if(neighbors < h_pairs2Count[k]){
					neighbors = h_pairs2Count[k];
				}
			}
	
			printf("%ld\t%d\t%d\t%d\t%d\t%d\t%d\n", step, countAlive, countRes, countNotres, countDead, countDrivers, neighbors);
			fprintf(out, "%ld\t%d\t%d\t%d\t%d\t%d\t%d\n", step, countAlive, countRes, countNotres, countDead, countDrivers, neighbors);
		}
	}
	fclose(out);

}

__global__ void division_death_kernel(float3* d_r, int* d_type, int* d_drivers, int* d_N, int N, int* d_pairsCount){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){
		if(randht(i) < 0.1){
			int drivers = d_drivers[i];
			int typei = d_type[i];
			if(typei != 0){
				float fit = ps*(drivers+1) - psigma*(2-typei) - pc*(typei-1) - pres*d_pairsCount[i];
				float p = randht(i);
				if(p < 0.8 + fit){
					int j = atomicAdd(d_N, 1);
					float3 ri = d_r[i];
					float3 rj;
					float4 dr = rforce(i);
					rj.x = ri.x + 0.3*dr.x;
					rj.y = ri.y + 0.3*dr.y;
					rj.z = ri.z + 0.3*dr.z;
					d_r[j] = rj;

					float p2 = randht(i);
					if(p2 < pu && drivers <= maxDrivers){
						drivers ++;
					}
					d_drivers[j] = drivers;
					p2 = randht(i);
					if(p2 < pv){
						typei = 2;
					}
					d_type[j] = typei;
				} else {
					d_type[i] = 3;
				}
			}
		}
	}
}

__global__ void repulsion_kernel(float3* d_r, float3* d_f, int* d_type, int N){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){
		int typei = d_type[i];
		if(typei > 0){
			float3 ri = d_r[i];
			float3 fi = d_f[i];
			int j;
			for(j = 0; j < N; j++){
				if(j != i){
					int typej = d_type[j];
					if(typej > 0){
						float3 rj = d_r[j];
						float dx = rj.x - ri.x;
						float dy = rj.y - ri.y;
						float dz = rj.z - ri.z;
						float dr = sqrtf(dx*dx + dy*dy + dz*dz);
						if(dr < 1.0f){
							float mult = 10000.0*(dr - 1.0f)/dr;
							fi.x += mult*dx;
							fi.y += mult*dy;
							fi.z += mult*dz;
						}
					}
				}
			}
			d_f[i] = fi;			
		}
	}
}

__global__ void integrate_kernel(float3* d_v, float3* d_r, float3* d_f, float* d_m, int* d_type, int N, float var){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){
		
		int typei = d_type[i];
		if(typei > 0 && typei != 3){

			float4 rf = rforce(i);

			float3 f = d_f[i];
			float3 r = d_r[i];
			float3 v = d_v[i];
			float m = d_m[i];

			f.x += -gamma*v.x + var*rf.x;
			f.y += -gamma*v.y + var*rf.y;
			f.z += -gamma*v.z + var*rf.z;		

			v.x += tau*f.x/m;
			v.y += tau*f.y/m;
			v.z += tau*f.z/m;

			r.x += v.x*tau;
			r.y += v.y*tau;
			r.z += v.z*tau;

			f.x = 0.0;
			f.y = 0.0;
			f.z = 0.0;

			d_r[i] = r;
			d_v[i] = v;	
			d_f[i] = f;
		}
	}
}

/*__global__ void pairlistGPU(float3* r, int* d_pairsCount, int* d_pairs, int* d_pairs2Count, int* d_pairs2, float cutoff, int N){
	int i, j, p;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 ri = r[i];
		int pairsCount = 0;
		for(p = 0; p < d_pairs2Count[i]; p++){
			j = d_pairs2[p*N + i];			
			float3 rj = r[j];
			if(getDistance(ri, rj) < cutoff) {
				d_pairs[pairsCount*N + i] = j;
				pairsCount ++;
			}
		}
		d_pairsCount[i] = pairsCount;
	}				
}
*/
__global__ void pairlist2GPU(float3* r, int* d_pairs2Count, int* d_pairs2, float cutoff2, int N){
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		float3 ri = r[i];
		int pairsCount = 0;
		for(j = 0; j < N; j++){
			if (j != i){
				float3 rj = r[j];
				if(getDistance(ri, rj) < cutoff2 /*&& pairsCount < 100*/) {
//					d_pairs2[pairsCount*maxN + i] = j;
					pairsCount ++;
				}
			}
		}
		d_pairs2Count[i] = pairsCount;
	}				
}

void saveCoord(long int step){
	int i;

	//printf("Step: %ld\t%f\t%f\t%f\n", step, h_r[0].x, h_r[0].y, h_r[0].z);
	cudaMemcpy(h_r, d_r, maxN*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_type, d_type, maxN*sizeof(int), cudaMemcpyDeviceToHost);
	int nonResistId = 0;
	int resistId = maxN;
	int deadID = 2*maxN;
	for(i = 0; i < maxN; i++){
		if(h_type[i] == 1){		
			dcd.frame.X[nonResistId] = 10.0*h_r[i].x;
			dcd.frame.Y[nonResistId] = 10.0*h_r[i].y;
			dcd.frame.Z[nonResistId] = 10.0*h_r[i].z;
			nonResistId ++;
		} else if(h_type[i] == 2){
			dcd.frame.X[resistId] = 10.0*h_r[i].x;
			dcd.frame.Y[resistId] = 10.0*h_r[i].y;
			dcd.frame.Z[resistId] = 10.0*h_r[i].z;
			resistId ++;
		} else if(h_type[i] == 3){
			dcd.frame.X[deadID] = 10.0*h_r[i].x;
			dcd.frame.Y[deadID] = 10.0*h_r[i].y;
			dcd.frame.Z[deadID] = 10.0*h_r[i].z;
			deadID ++;
		}
	}
	for(i = nonResistId; i < maxN; i++){
		dcd.frame.X[i] = 0.0;
		dcd.frame.Y[i] = 0.0;
		dcd.frame.Z[i] = 0.0;
	}
	for(i = resistId; i < 2*maxN; i++){
		dcd.frame.X[i] = 0.0;
		dcd.frame.Y[i] = 0.0;
		dcd.frame.Z[i] = 0.0;
	}
	for(i = deadID; i < 3*maxN; i++){
		dcd.frame.X[i] = 0.0;
		dcd.frame.Y[i] = 0.0;
		dcd.frame.Z[i] = 0.0;
	}
	dcdWriteFrame(dcd);
}
