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
//    size_t f1[2] = {msh.vtx_dim.y, msh.vtx_dim.z};
    
    //kernels
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, nv, NULL, 0, NULL, NULL); //&ocl.event
    
    //for profiling
//    clWaitForEvents(1, &ocl.event);
    
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd1, 3, NULL, nv, NULL, 0, NULL, NULL); //c
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.fac_bnd1, 2, NULL, f1, NULL, 0, NULL, NULL); //u
    

    //read from device
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.vtx_xx, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.vtx_xx, 0, NULL, NULL);

    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U0, CL_TRUE, 0, 4*msh.nv_tot*sizeof(float), ocl.hst.U0, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U1, CL_TRUE, 0, 4*msh.nv_tot*sizeof(float), ocl.hst.U1, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.F1, CL_TRUE, 0, 4*msh.nv_tot*sizeof(float), ocl.hst.F1, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.J.ii, CL_TRUE, 0, 27*16*msh.nv_tot*sizeof(int),   ocl.hst.J.ii, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.J.jj, CL_TRUE, 0, 27*16*msh.nv_tot*sizeof(int),   ocl.hst.J.jj, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.J.vv, CL_TRUE, 0, 27*16*msh.nv_tot*sizeof(float), ocl.hst.J.vv, 0, NULL, NULL);

    
    //reset
//    memset(ocl.hst.U1u, 0, 3*msh.nv_tot*sizeof(float));
//    memset(ocl.hst.U1c, 0, 1*msh.nv_tot*sizeof(float));
    
//    //solve
//    slv_mtx(&msh, &ocl);
    
    //store prior
//    ocl.err = clEnqueueCopyBuffer( ocl.command_queue, ocl.dev.U1c, ocl.dev.U0c, 0, 0, 1*msh.nv_tot*sizeof(float), 0, NULL, NULL);
    
    //write to device
//    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.dev.U1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.U1u, 0, NULL, NULL);
//    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.dev.U1c, CL_TRUE, 0, 1*msh.nv_tot*sizeof(float), ocl.hst.U1c, 0, NULL, NULL);
     
    
//    write vtk
    wrt_vtk(&msh, &ocl);
    
    //write for matlab
    wrt_raw(ocl.hst.J.ii, 27*16*msh.nv_tot, sizeof(int),   "J_ii");
    wrt_raw(ocl.hst.J.jj, 27*16*msh.nv_tot, sizeof(int),   "J_jj");
    wrt_raw(ocl.hst.J.vv, 27*16*msh.nv_tot, sizeof(float), "J_vv");
    
    wrt_raw(ocl.hst.U0, 4*msh.nv_tot, sizeof(float), "U0");
    wrt_raw(ocl.hst.U1, 4*msh.nv_tot, sizeof(float), "U1");
    wrt_raw(ocl.hst.F1, 4*msh.nv_tot, sizeof(float), "F1");

    //clean
    ocl_final(&msh, &ocl);
    
    printf("done\n");
    
    return 0;
}
