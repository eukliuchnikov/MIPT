typedef struct {
	int N;
	float3* h_r;
	float3* h_v;
	float3* h_f;
	float* h_m;
	float3* d_r;
	float3* d_v;
	float3* d_f;
	float* d_m;
	int* h_type;
	int* d_type;
} MDSystem;

typedef struct {
	int i, j;
	float r0, ks;
} HarmonicPair;

int harmonicCount;
HarmonicPair* harmonicPairs;
