#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "pdbio.h"
#include "xyzio.h"
#include "ran2.h"


typedef struct {
	float x, y, z;
} float3;

int N = 5000;
float Lx = 300.0f;
float Ly = 300.0f;
float Lz = 300.0f;
float r0 = 1.0;
float T = 300.0f;
float m = 1.0;
float sigma;
float epsilon;

float dt = 0.001f;
int totalsteps = 10000;
int stride = 10;

#define kB 8.314462e-3 //Boltzman const

float3* r;
float3* v;
float3* f;
int rseed = -738554;

float len(float3 a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

float3 getVector(float3 ri, float3 rj, float Lx, float Ly, float Lz){
	float3 dr;	
	dr.x = rj.x - ri.x;
	dr.y = rj.y - ri.y;
	dr.z = rj.z - ri.z;
	dr.x -= rint(dr.x/Lx)*Lx;
	dr.y -= rint(dr.y/Ly)*Ly;
	dr.z -= rint(dr.z/Lz)*Lz;
	return dr;
}

float getDistance(float3 ri, float3 rj, float Lx, float Ly, float Lz){
	float3 dr = getVector(ri, rj, Lx, Ly, Lz);
	return len(dr);
}

void placeAtoms(){
	float3 r1;
	int i, j;
	for(i = 0; i < N; i++){
		float mindist = 0.0;
		while(mindist < 4.0){
			r1.x = Lx*(0.5 - ran2(&rseed)); 
			r1.y = Ly*(0.5 - ran2(&rseed)); 
			r1.z = Lz*(0.5 - ran2(&rseed));

			mindist = FLT_MAX;
			for(j = 0; j < i; j++){
				float dist = getDistance(r1, r[j], Lx, Ly, Lz);
				if(dist < mindist){
					mindist = dist;
				}
			}
		}
		r[i] = r1;
		printf("Placing %d\n", i);
	}
}


int writeXYZ(const char* filename, float3* data, int N, const char* modifier){
	FILE* file = fopen(filename, modifier);
	fprintf(file, "%d\n", N);
	fprintf(file, "Coor\n");
	int i;
	for (i = 0; i < 0.5*N; i++){
		fprintf(file, "0\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
	for (i = 0.5*N; i < N; i++){
		fprintf(file, "1\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
/*	if(i == 29){
		fprintf(file, "5\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
	for (i = 30; i < 120; i++){
		fprintf(file, "1\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
	for (i = 120; i < 290; i++){
		fprintf(file, "2\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
	for (i = 290; i < 1690; i++){
		fprintf(file, "3\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
	for (i = 1690; i < N; i++){
		fprintf(file, "4\t%f\t%f\t%f\n", 10.0*data[i].x, 10.0*data[i].y, 10.0*data[i].z);
	}
*/	fclose(file);
}


int main(int argc, char* argv[]){
	r = (float3*)calloc(N, sizeof(float3));
	v = (float3*)calloc(N, sizeof(float3));
	placeAtoms();
	writeXYZ("Atomscoor.xyz", r, N, "w");
	return 1;
}

