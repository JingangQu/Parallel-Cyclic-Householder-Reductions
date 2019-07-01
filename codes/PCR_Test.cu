#include <stdio.h>
#include <stdlib.h>
#include "PCR_Solver.cuh"
#include "Householder.cuh"


// I refer to the "testCUDA" of Lokman A. Abbas-Turki
// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// "count_iter" is used to calculate the maximum number of iterations allowed
int count_iter(int bin) {

	int m = 0;
	int count = 0;

	while (bin != 0) {
		m+=bin%2;
		bin = bin >> 1;
		count++;
	}
	count--;

	if (m>1) m = 0;

    return (count - m);
}

// This function can generate a random symmetric matrix
void rand_sym_matrix(float *a, float *dlist, int dim) {

	srand((unsigned) time(NULL));
	for(int i=0;i<dim;++i) {
	    dlist[i] = rand()%100 + 0.15;
		for(int j=0;j<dim;++j)
		{
			if(i<j) a[i*dim + j] = rand()%50 + 1.5;
			else if(i>j) a[i*dim + j] = a[j*dim + i];
			else a[i*dim + j] = rand()%50 + 1.5;
		}
	}

	printf("\n This is the random symmetric matrix A and the random second member d: \n");
	for(int i=0;i<dim;++i) {
	    for(int j=0;j<dim;++j)
			   printf("%10f",a[i*dim + j]);

	    printf("    %10f",dlist[i]);
	    printf("\n");}
}


int main( ) {
	
	int N = 11;  // diagonal size
	
	float *clist, *dlist, *xlist, *A, *D, *S;
    float *dev_c, *dev_d, *dev_x, *dev_A, *dev_D, *dev_S;
	
	// allocate the memory on the CPU
	A = (float*)malloc( N * N * sizeof(float) );
	D = (float*)malloc( N * sizeof(float) );
	S = (float*)malloc( N * sizeof(float) );
    clist = (float*)malloc( N * sizeof(float) );
	dlist = (float*)malloc( N * sizeof(float) );
	xlist = (float*)malloc( N * sizeof(float) );
	
	// allocate the memory on the GPU
	testCUDA( cudaMalloc( (void**)&dev_A, N * N * sizeof(float) ) );
	testCUDA( cudaMalloc( (void**)&dev_D, N * sizeof(float) ) );
    testCUDA( cudaMalloc( (void**)&dev_S, N * sizeof(float) ) );
    testCUDA( cudaMalloc( (void**)&dev_c, N * sizeof(float) ) );
	testCUDA( cudaMalloc( (void**)&dev_d, N * sizeof(float) ) );
	testCUDA( cudaMalloc( (void**)&dev_x, N * sizeof(float) ) );

	rand_sym_matrix(A, dlist, N);
	// copy the arrays A to the GPU

    testCUDA( cudaMemcpy( dev_A, A, N * N * sizeof(float),
                              cudaMemcpyHostToDevice ) );

	tred_fact_k<<<1,1,1*N*(N+2)*sizeof(float)>>>(dev_A, dev_D, dev_S, N);
	
	// copy the array back from the GPU to the CPU
    testCUDA( cudaMemcpy( A, dev_A, N * N * sizeof(float),
                          cudaMemcpyDeviceToHost ) );
    testCUDA( cudaMemcpy( D, dev_D, N * sizeof(float),
                          cudaMemcpyDeviceToHost ) );
    testCUDA( cudaMemcpy( S, dev_S, N * sizeof(float),
                          cudaMemcpyDeviceToHost ) );
    printf("\n This is the orthogonal matrix Q effecting the transformation in householder: \n");
    for(int i=0;i<N;++i) {
        for(int j=0;j<N;++j)
            printf("%10f",A[i*N + j]);
       printf("\n");
    }
    printf("\n This is the diagonal elements of the tridiagonal matrix in householder: \n");
    for(int i=0;i<N;++i) {
        printf("%10f",D[i]);
        printf("\n");
    }
    printf("\n This is the off-diagonal elements of the tridiagonal matrix in householder: \n");
    for(int i=0;i<N;++i) {
        printf("%10f",S[i]);
        printf("\n");
    }

	int iter_max;
	iter_max = count_iter(N);
	printf("\n The maximum number of PCR iteration is: %d \n", iter_max);

	for (int it=0; it<N; it++) {
	    if (it == N-1)
	        clist[it] = 0;
	    else
		    clist[it] = S[it+1];
    }

	// copy the arrays to the GPU
	testCUDA( cudaMemcpy( dev_c, clist, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    testCUDA( cudaMemcpy( dev_d, dlist, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	testCUDA( cudaMemcpy( dev_x, xlist, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
			
	// Solve_Kernel(Q, alist, blist, clist, dlist, xlist, iter_max, DMax)
	// Q = dev_A, dev_S = alist, dev_D = blist, dev_c = clist
	Solve_Kernel<<<1,N>>>(dev_A, dev_S, dev_D, dev_c, dev_d, dev_x, iter_max, N);

	// copy the array back from the GPU to the CPU
	testCUDA( cudaMemcpy( xlist, dev_x, N * sizeof(float),
						  cudaMemcpyDeviceToHost ) );

	printf("\n The final resolution is : \n");
    for (int it=0; it<N; it++) {
		printf( "%10f\n", xlist[it] );
    }
	
	// free the memory we allocated on the GPU
	testCUDA( cudaFree( dev_A ) );
    testCUDA( cudaFree( dev_D ) );
    testCUDA( cudaFree( dev_S ) );
    testCUDA( cudaFree( dev_c ) );
	testCUDA( cudaFree( dev_d ) );
    testCUDA( cudaFree( dev_x ) );

    // free the memory we allocated on the CPU
	free( A );
	free( D );
	free( S );
    free( clist );
	free( dlist );
    free( xlist );
	
	
    return 0;

}





