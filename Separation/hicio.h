#pragma once

typedef struct {
	int i;
	int j;
	int n;
} HiCPair;
 
typedef struct {
	int atomCount;
	int pairCount;
	HiCPair* pairs;
} HiCData;

void readHiC(const char* filename, HiCData* hic);
void writeHiC(const char* filename, HiCData* hic);
