//
//  main.c
//  frac2
//
//  Created by Toby Simpson on 12.01.24.
//


#include <stdio.h>
#include <OpenCL/opencl.h>
#include <Accelerate/Accelerate.h>

#include "msh.h"
#include "ocl.h"
#include "slv.h"
#include "io.h"

//for later
//clSetKernelArg(myKernel, 0, sizeof(cl_int), &myVariable).

//here
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //objects
    struct msh_obj msh;
    struct ocl_obj ocl;
    
    //init obj
    msh_init(&msh);
    ocl_init(&msh, &ocl);
    
    //cast dims
    size_t nv[3] = {msh.vtx_dim.x, msh.vtx_dim.y, msh.vtx_dim.z};
    
    //init
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, nv, NULL, 0, NULL, NULL);
    
    //bnd1, notch
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd1, 3, NULL, nv, NULL, 0, NULL, NULL);
    
    //bnd2 - displacement U
    msh.mat_prm.s7 = 1e-1f;                                                                                      //update
    ocl.err = clSetKernelArg(ocl.vtx_bnd2,  1, sizeof(cl_float8), (void*)&msh.mat_prm);                         //refresh
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd2, 3, NULL, nv, NULL, 0, NULL, NULL);
    
    //copy U1 to U0
    ocl.err = clEnqueueCopyBuffer( ocl.command_queue, ocl.dev.U1, ocl.dev.U0, 0, 0, 4*msh.nv_tot*sizeof(float), 0, NULL, NULL);  //copy to prev
    
    //assemble
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, nv, NULL, 0, NULL, &ocl.event);
    clWaitForEvents(1, &ocl.event);                                                                             //for profiling
    
    //bnd3 - displacement F, identity
    ocl.err = clSetKernelArg(ocl.vtx_bnd3,  1, sizeof(cl_float8), (void*)&msh.mat_prm);                         //refresh
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd3, 3, NULL, nv, NULL, 0, NULL, NULL);
    
    //bnd4 crack I
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd4, 3, NULL, nv, NULL, 0, NULL, NULL);

    //read from device
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.vtx_xx, CL_TRUE, 0, msh.nv_tot*sizeof(cl_float4), ocl.hst.vtx_xx, 0, NULL, NULL);

    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U0, CL_TRUE, 0, msh.nv_tot*sizeof(cl_float4), ocl.hst.U0, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U1, CL_TRUE, 0, msh.nv_tot*sizeof(cl_float4), ocl.hst.U1, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.F1, CL_TRUE, 0, msh.nv_tot*sizeof(cl_float4), ocl.hst.F1, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.J.ii, CL_TRUE, 0, 27*msh.nv_tot*sizeof(cl_int16),   ocl.hst.J.ii, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.J.jj, CL_TRUE, 0, 27*msh.nv_tot*sizeof(cl_int16),   ocl.hst.J.jj, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.J.vv, CL_TRUE, 0, 27*msh.nv_tot*sizeof(cl_float16), ocl.hst.J.vv, 0, NULL, NULL);

    
    //reset
//    memset(ocl.hst.U1u, 0, 3*msh.nv_tot*sizeof(float));
//    memset(ocl.hst.U1c, 0, 1*msh.nv_tot*sizeof(float));
    
    //solve
//    slv_mtx(&msh, &ocl);
    
    //store prior
//    ocl.err = clEnqueueCopyBuffer( ocl.command_queue, ocl.dev.U1c, ocl.dev.U0c, 0, 0, 1*msh.nv_tot*sizeof(float), 0, NULL, NULL);
    
    //write to device
//    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.dev.U1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.U1u, 0, NULL, NULL);
//    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.dev.U1c, CL_TRUE, 0, 1*msh.nv_tot*sizeof(float), ocl.hst.U1c, 0, NULL, NULL);
     
    
//    write vtk
    wrt_vtk(&msh, &ocl);
    
    //write for matlab
    wrt_raw(ocl.hst.J.ii, 27*msh.nv_tot, sizeof(cl_int16),   "J_ii");
    wrt_raw(ocl.hst.J.jj, 27*msh.nv_tot, sizeof(cl_int16),   "J_jj");
    wrt_raw(ocl.hst.J.vv, 27*msh.nv_tot, sizeof(cl_float16), "J_vv");
    
    wrt_raw(ocl.hst.U0, msh.nv_tot, sizeof(cl_float4), "U0");
    wrt_raw(ocl.hst.U1, msh.nv_tot, sizeof(cl_float4), "U1");
    wrt_raw(ocl.hst.F1, msh.nv_tot, sizeof(cl_float4), "F1");
    
    
    //profile
    cl_ulong time0;
    cl_ulong time1;

    clGetEventProfilingInfo(ocl.event, CL_PROFILING_COMMAND_START, sizeof(time0), &time0, NULL);
    clGetEventProfilingInfo(ocl.event, CL_PROFILING_COMMAND_END,   sizeof(time1), &time1, NULL);

    double nanoSeconds = time1-time0;
    printf("nv, time(ms)\n");
    printf("%07d, %0.4f;\n", msh.nv_tot, nanoSeconds/1000000.0);

    //clean
    ocl_final(&msh, &ocl);
    
    printf("done\n");
    
    return 0;
}
