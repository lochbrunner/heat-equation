

#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h


texture<float4, 2, cudaReadModeElementType> inTex;
texture<float4, 2, cudaReadModeElementType> boundaryTex;


__device__ float length2(float3 &a)
{
	return a.x*a.x+a.y*a.y+a.z*a.z;
}

__device__ float mul(float3 &a, float3 &b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z;
}


__global__ void
cudaProcess(float4* g_odata, int imgw, int imgh, 
	    int tilew, float dw, float dh, float speed)

{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;
		
	
	// Declare values
	float4 boundary = tex2D(boundaryTex, x, y);
	float4 v00 = tex2D(inTex, x, y);
	float4 result = v00;
	float4 v0p;
	float4 vp0;
	float4 v0m;
	float4 vm0;
	// Define values
	if(x!=imgw-1) v0p =  tex2D(inTex, x+1, y);
	else v0p =  v00;
	if(y!=imgh-1) vp0 =  tex2D(inTex, x, y+1);
	else vp0 =  v00;
	if(x!=0) v0m =  tex2D(inTex, x-1, y);
	else v0m =  v00;
	if(y!=0) vm0 =  tex2D(inTex, x, y-1);
	else vm0 =  v00;

	__syncthreads();

	// calculate new value
	result.x = (v0p.x*v0p.y + v0m.x*v0m.y + vp0.x*vp0.y + vm0.x*vm0.y)*v00.y*speed + (1.0f - v00.y*(v0p.y+v0m.y+vp0.y+vm0.y)*speed)*v00.x;
	result.x = result.x *(1.0f - boundary.z) + boundary.z*boundary.x;
	// calculate normals
	result.z = (v0p.x-v0m.x)/(dw*2.0f);
	result.w = (vp0.x-vm0.x)/(dh*2.0f);
	
	g_odata[y*imgw+x] = result;
}



__global__ void
cudaPicking(float4* g_odata, int imgw, int imgh, 
	    int tilew, float dw, float dh, float3 orig, float3 dir, float value)

{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

	float fw = imgw;
	float fh = imgh;

	float pos_x =  2.0f*2.0f/(fw -1.0f)*x -2.0f;
	float pos_z =  2.0f*2.0f/(fh -1.0f)*y -2.0f; 
		
	
	float4 v00 = tex2D(inTex, x, y);
	float4 result = v00;
		
	float3 pos;
	pos.x = pos_x - orig.x;
	pos.y = v00.x - orig.y;
	pos.z = pos_z - orig.z;

	// lot = pos - dir*lambda
	// dir * lot = 0
	// => dir * (pos - dir*lambda) = 0
	// => pos*dir = dir*dir*lambda;
	// => lambda = pos*dir / dir*dir;
	float lambda = mul(pos, dir) / mul(dir, dir);
	float3 lot;
	lot.x = pos.x - lambda*dir.x;
	lot.y = pos.y - lambda*dir.y;
	lot.z = pos.z - lambda*dir.z;

	float dis2 = length2(lot);

	result.x = v00.x + value*expf(-dis2*16.0f);

	g_odata[y*imgw+x] = result;
}


extern "C"
void init_boundary_cuda(cudaArray *boundary_array)
{
	cutilSafeCall(cudaBindTextureToArray(boundaryTex, boundary_array));

	struct cudaChannelFormatDesc desc; 
    cutilSafeCall(cudaGetChannelDesc(&desc, boundary_array));
}



extern "C"
void launch_cudaProcess(dim3 grid, dim3 block, int sbytes, 
		   cudaArray *g_data_array, float4* g_odata, 
		   int imgw, int imgh, int tilew, float dw, float dh, float speed)
{
	cutilSafeCall(cudaBindTextureToArray(inTex, g_data_array));

	struct cudaChannelFormatDesc desc; 
    cutilSafeCall(cudaGetChannelDesc(&desc, g_data_array));


	cudaProcess<<< grid, block, sbytes >>> (g_odata, imgw, imgh, 
			    block.x, dw, dh, speed);
}




extern "C"
void launch_cudaPicking(dim3 grid, dim3 block, int sbytes, 
		   cudaArray *g_data_array, float4* g_odata, 
		   int imgw, int imgh, int tilew, float dw, float dh, float3 orig, float3 dir, float value)
{
	cutilSafeCall(cudaBindTextureToArray(inTex, g_data_array));

	struct cudaChannelFormatDesc desc; 
    cutilSafeCall(cudaGetChannelDesc(&desc, g_data_array));


	cudaPicking<<< grid, block, sbytes >>> (g_odata, imgw, imgh, 
			    block.x, dw, dh, orig, dir, value);
}