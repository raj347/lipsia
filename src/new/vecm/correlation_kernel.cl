#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void correlationKernel(const __global float * const mat, __global float* A, const unsigned int nt, unsigned int type, long n)
{
	const float tiny=1.0e-4;
	const unsigned int id1 = get_global_id(0);
	const unsigned int id2 = get_global_id(1);
	const unsigned int id = id1 + id2 * 512;
	if( id < n ) {
		for (unsigned int j=0; j<=id; j++) {
			const unsigned int k=j+id*(id+1)/2;
			A[k] = 0;
			if( j != id ) {
				double sx,sy,sxx,syy,sxy;
				sx = 0;
				sy = 0;
				sxx = 0;
				syy = 0;
				sxy = 0;
				for( unsigned int i=0;i<nt;i++ ) {
					const float v1 = mat[i+nt*id];
					const float v2 = mat[i+nt*j];
					sx += v1;
					sy += v2;
					sxx += v1*v1;
					syy += v2*v2;
					sxy += v1*v2;
				}
				const double u = nt * sxx - sx * sx;
				const double v = nt * syy - sy * sy;
				if((u*v) > tiny ) {
					A[k] = (nt*sxy - sx*sy)/sqrt(u*v);
				}
				if(type == 1 ) {
					A[k] += 1.0;
				}
				if( type == 2 ) {
					A[k] = fabs(A[k]);
				}
				if( A[k] < tiny ) {
					A[k] = tiny;
				}
			}
		}
	}
}