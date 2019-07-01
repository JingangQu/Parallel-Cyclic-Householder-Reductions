// This file is provided by the teacher

#ifndef Householder_cuh
#define Householder_cuh

/**********************************************************************
Code associated to the paper:

Resolution of a large number of small random symmetric 
linear systems in single precision arithmetic on GPUs

by: 

Lokman A. Abbas-Turki and Stef Graillat

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include "Hcyc.h"

/**********************************************************************

Only factorization

**********************************************************************/
///////////////////////////////////////////////////////////////////////
// Tridiagonalization: one thread per matrix
// This is a straight modification of function tred2 in 
// Numerical Recipes in C
///////////////////////////////////////////////////////////////////////
#define EPS (0.000001f)


__global__ void tred_fact_k(float *a, float *d, float *s, int n)
{

	// The global memory access index
	int gb_index_x = threadIdx.x + blockIdx.x*blockDim.x;
	// Local integers
	int l, k, j, i, n2, nt;
	// Local floats
	float scale, hh, h, g, f;
	// Shared memory
	extern __shared__ float sAds[];

	n2 = n*n;
	nt = threadIdx.x*n*(n + 2);

	// Copy the matrix from global to shared memory
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			// sAds stores s, a, d, so nt = ...*n*(n+2)
			sAds[nt + i*n + j] = a[gb_index_x*n*n + i*n + j];
		}
	}

	// Computing the tridiagonal form U.
	for (i = n - 1; i >= 1; i--) {
		l = i - 1;
		h = scale = 0.0f;
		if (l > 0) {
			for (k = 0; k <= l; k++)
				scale += fabs(sAds[nt + i*n + k]);
			if (scale < EPS) //Skip transformation.
				s[gb_index_x*n + i] = sAds[nt + i*n + l];
			else {
				for (k = 0; k <= l; k++) {
					sAds[nt + i*n + k] /= scale; //Use scaled A�s for transformation.
					h += sAds[nt + i*n + k] * sAds[nt + i*n + k]; //Compute sigma.
				}
				f = sAds[nt + i*n + l];
				g = (f >= 0.0f ? -sqrtf(h) : sqrtf(h));
				s[gb_index_x*n + i] = scale*g;
				h -= f*g; //Compute b.
				sAds[nt + i*n + l] = f - g; //Store u in the ith row of A.
				f = 0.0f;
				for (j = 0; j <= l; j++) {
					sAds[nt + j*n + i] = sAds[nt + i*n + j] / h; //Store u/b in the ith column of A.
					g = 0.0f; //Form an element of A�u in g.
					for (k = 0; k <= j; k++)
						g += sAds[nt + j*n + k] * sAds[nt + i*n + k];
					for (k = j + 1; k <= l; k++)
						g += sAds[nt + k*n + j] * sAds[nt + i*n + k];
					sAds[nt + n2 + n + j] = g / h; //Form element of p in temporarily unused
					//element of e.
					f += sAds[nt + n2 + n + j] * sAds[nt + i*n + j];
				}
				hh = f / (h + h); //Compute B.
				for (j = 0; j <= l; j++) { //Form q and store it in e overwriting p.
					f = sAds[nt + i*n + j];
					sAds[nt + n2 + n + j] = g = sAds[nt + n2 + n + j] - hh*f;
					for (k = 0; k <= j; k++) //Reduce A.
						sAds[nt + j*n + k] -= (f*sAds[nt + n2 + n + k] + g*sAds[nt + i*n + k]);
				}
			}
		}
		else
			s[gb_index_x*n + i] = sAds[nt + i*n + l];
		sAds[nt + n2 + i] = h;
	}

	sAds[nt + n2] = 0.0f;
	s[gb_index_x*n] = 0.0f;

	//Accumulation of transformation matrices to get the product matrix Q.
	for (i = 0; i <= n - 1; i++) {
		l = i - 1;
		if (sAds[nt + n2 + i]) { //This block skipped when i=0.
			for (j = 0; j <= l; j++) {
				g = 0.0f;
				for (k = 0; k <= l; k++) //Use u and u/b stored in A to form H�Q.
					g += sAds[nt + i*n + k] * sAds[nt + k*n + j];
				for (k = 0; k <= l; k++)
					sAds[nt + k*n + j] -= g*sAds[nt + k*n + i];
			}
		}
		d[gb_index_x*n + i] = sAds[nt + i*n + i];
		sAds[nt + i*n + i] = 1.0f; //Reset row and column of A to identity matrix for ...
		for (j = 0; j <= l; j++) sAds[nt + j*n + i] = sAds[nt + i*n + j] = 0.0f; // ... next iteration.
	}
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			a[gb_index_x*n*n + i*n + j] = sAds[nt + i*n + j];
		}
	}
}

#endif
