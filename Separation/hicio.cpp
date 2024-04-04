#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hicio.h"

#define BUF_SIZE 100000

void addPair(int i, int j, int n, HiCData* hic){
	
	if(i > j && i != j && n > 0){
		HiCPair pair;
		pair.i = i;
		pair.j = j;
		pair.n = n;
		//printf("Adding: %d - %d - %d \n", i, j, n);
		hic->pairs[hic->pairCount] = pair;
		hic->pairCount ++;
	}
}

void readHiC(const char* filename, HiCData* hic){
	printf("Reading %s.\n", filename);
	char buffer[BUF_SIZE];
	FILE* file = fopen(filename, "r");
	if(file != NULL){
		int atomCount = 0;
		while(fgets(buffer, BUF_SIZE, file) != NULL){
			atomCount ++;
		}
		hic->atomCount = atomCount;
		rewind(file);

		int i = 0;
		int j = 0;
		hic->pairCount = 0;
		while(fgets(buffer, BUF_SIZE, file) != NULL){
			char* pch;
			pch = strtok(buffer, " \t\r\n");
			j = 0;
			hic->pairCount ++;
			for(j = 1; j < atomCount; j++){
				pch = strtok(NULL, " \t\r\n");
				hic->pairCount ++;
			}

			i ++;
		}
		hic->pairs = (HiCPair*)calloc(hic->pairCount, sizeof(HiCPair));
		rewind(file);
		i = 0;
		j = 0;
		hic->pairCount = 0;
		while(fgets(buffer, BUF_SIZE, file) != NULL){
			char* pch;
			pch = strtok(buffer, " \t\r\n");
			j = 0;
			addPair(i, j, atoi(pch), hic);
			for(j = 1; j < atomCount; j++){
				pch = strtok(NULL, " \t\r\n");
				addPair(i, j, atoi(pch), hic);
			}

			i ++;
		}		


		printf("Done reading '%s'.\n", filename);
		fclose(file);
	} else {
		perror(filename);
		exit(0);
	}
}

void writeHiC(const char* filename, HiCData* hic){
	//TODO
}
