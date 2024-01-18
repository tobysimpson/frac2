//
//  ocl.h
//  frac2
//
//  Created by Toby Simpson on 12.01.24.
//

#ifndef ocl_h
#define ocl_h


#define ROOT_PRG    "/Users/toby/Documents/USI/postdoc/fracture/xcode/frac2/frac2"

struct coo_dev
{
    cl_mem  ii;
    cl_mem  jj;
    cl_mem  vv;
};

struct coo_hst
{
    int     *ii;
    int     *jj;
    float   *vv;
};

struct mem_hst
{
    cl_float4*  vtx_xx;
    
    cl_float4*  U0;     //prev
    cl_float4*  U1;     //sln
    cl_float4*  F1;     //rhs
    
    struct coo_hst J;
};

struct mem_dev
{
    cl_mem vtx_xx;

    cl_mem U0;
    cl_mem U1;
    cl_mem F1;

    struct coo_dev J;
};

//object
struct ocl_obj
{
    //environment
    cl_int              err;
    cl_platform_id      platform_id;
    cl_device_id        device_id;
    cl_uint             num_devices;
    cl_uint             num_platforms;
    cl_context          context;
    cl_command_queue    command_queue;
    cl_program          program;
    char                device_str[100];
    cl_event            event;  //for profiling
        
    //memory
    struct mem_hst hst;
    struct mem_dev dev;
    
    //kernels
    cl_kernel           vtx_init;
    cl_kernel           vtx_assm;
    cl_kernel           vtx_bnd1;
    cl_kernel           vtx_bnd2;
    cl_kernel           vtx_bnd3;
    cl_kernel           vtx_bnd4;
};


//init
void ocl_init(struct msh_obj *msh, struct ocl_obj *ocl)
{
    printf("__FILE__: %s\n", __FILE__);
    
    /*
     =============================
     environment
     =============================
     */
    
    ocl->err            = clGetPlatformIDs(1, &ocl->platform_id, &ocl->num_platforms);                                              //platform
    ocl->err            = clGetDeviceIDs(ocl->platform_id, CL_DEVICE_TYPE_GPU, 1, &ocl->device_id, &ocl->num_devices);              //devices
    ocl->context        = clCreateContext(NULL, ocl->num_devices, &ocl->device_id, NULL, NULL, &ocl->err);                          //context
    ocl->command_queue  = clCreateCommandQueue(ocl->context, ocl->device_id, CL_QUEUE_PROFILING_ENABLE, &ocl->err);                 //command queue
    ocl->err            = clGetDeviceInfo(ocl->device_id, CL_DEVICE_NAME, sizeof(ocl->device_str), &ocl->device_str, NULL);         //device info
    
    printf("%s\n", ocl->device_str);
    
    /*
     =============================
     program
     =============================
     */
    
    //name
    char prg_name[200];
    sprintf(prg_name,"%s/%s", ROOT_PRG, "prg.cl");

    printf("%s\n",prg_name);

    //file
    FILE* src_file = fopen(prg_name, "r");
    if(!src_file)
    {
        fprintf(stderr, "Failed to load kernel. check ROOT_PRG\n");
        exit(1);
    }

    //length
    fseek(src_file, 0, SEEK_END);
    size_t  prg_len =  ftell(src_file);
    rewind(src_file);

//    printf("%lu\n",prg_len);

    //source
    char *prg_src = (char*)malloc(prg_len);
    fread(prg_src, sizeof(char), prg_len, src_file);
    fclose(src_file);

//    printf("%s\n",prg_src);

    //create
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&prg_src, (const size_t*)&prg_len, &ocl->err);
    printf("prg %d\n",ocl->err);

    //build
    ocl->err = clBuildProgram(ocl->program, 1, &ocl->device_id, NULL, NULL, NULL);
    printf("bld %d\n",ocl->err);

    //log
    size_t log_size = 0;
    
    //log size
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    //allocate
    char *log = (char*)malloc(log_size);

    //log text
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    //print
    printf("%s\n", log);

    //clear
    free(log);

    //clean
    free(prg_src);

    //unload compiler
    ocl->err = clUnloadPlatformCompiler(ocl->platform_id);
    
    /*
     =============================
     kernels
     =============================
     */

    ocl->vtx_init = clCreateKernel(ocl->program, "vtx_init", &ocl->err);
    ocl->vtx_assm = clCreateKernel(ocl->program, "vtx_assm", &ocl->err);
    ocl->vtx_bnd1 = clCreateKernel(ocl->program, "vtx_bnd1", &ocl->err);
    ocl->vtx_bnd2 = clCreateKernel(ocl->program, "vtx_bnd2", &ocl->err);
    ocl->vtx_bnd3 = clCreateKernel(ocl->program, "vtx_bnd3", &ocl->err);
    ocl->vtx_bnd4 = clCreateKernel(ocl->program, "vtx_bnd4", &ocl->err);
    
    /*
     =============================
     memory
     =============================
     */
    
    //host
    ocl->hst.vtx_xx = malloc(msh->nv_tot*sizeof(cl_float4));
    
    ocl->hst.U0 = malloc(msh->nv_tot*sizeof(cl_float4));
    ocl->hst.U1 = malloc(msh->nv_tot*sizeof(cl_float4));
    ocl->hst.F1 = malloc(msh->nv_tot*sizeof(cl_float4));
    
    ocl->hst.J.ii = malloc(27*msh->nv_tot*sizeof(cl_int16));
    ocl->hst.J.jj = malloc(27*msh->nv_tot*sizeof(cl_int16));
    ocl->hst.J.vv = malloc(27*msh->nv_tot*sizeof(cl_float16));
    

    //CL_MEM_READ_WRITE/CL_MEM_HOST_READ_ONLY/CL_MEM_HOST_NO_ACCESS
    
    //device
    ocl->dev.vtx_xx = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    
    ocl->dev.U0   = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    ocl->dev.U1   = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE    , msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    ocl->dev.F1   = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    
    ocl->dev.J.ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(cl_int16),   NULL, &ocl->err);
    ocl->dev.J.jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(cl_int16),   NULL, &ocl->err);
    ocl->dev.J.vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(cl_float16), NULL, &ocl->err);
    

    /*
     =============================
     arguments
     =============================
     */

    ocl->err = clSetKernelArg(ocl->vtx_init,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_init,  1, sizeof(cl_float3), (void*)&msh->x0);
    ocl->err = clSetKernelArg(ocl->vtx_init,  2, sizeof(cl_float3), (void*)&msh->dx);
    ocl->err = clSetKernelArg(ocl->vtx_init,  3, sizeof(cl_mem),    (void*)&ocl->dev.vtx_xx);
    ocl->err = clSetKernelArg(ocl->vtx_init,  4, sizeof(cl_mem),    (void*)&ocl->dev.U0);
    ocl->err = clSetKernelArg(ocl->vtx_init,  5, sizeof(cl_mem),    (void*)&ocl->dev.U1);
    ocl->err = clSetKernelArg(ocl->vtx_init,  6, sizeof(cl_mem),    (void*)&ocl->dev.F1);
    ocl->err = clSetKernelArg(ocl->vtx_init,  7, sizeof(cl_mem),    (void*)&ocl->dev.J.ii);
    ocl->err = clSetKernelArg(ocl->vtx_init,  8, sizeof(cl_mem),    (void*)&ocl->dev.J.jj);
    ocl->err = clSetKernelArg(ocl->vtx_init,  9, sizeof(cl_mem),    (void*)&ocl->dev.J.vv);

    ocl->err = clSetKernelArg(ocl->vtx_assm,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  1, sizeof(cl_float3), (void*)&msh->dx);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  2, sizeof(cl_float8), (void*)&msh->mat_prm);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  3, sizeof(cl_mem),    (void*)&ocl->dev.U0);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  4, sizeof(cl_mem),    (void*)&ocl->dev.U1);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  5, sizeof(cl_mem),    (void*)&ocl->dev.F1);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  6, sizeof(cl_mem),    (void*)&ocl->dev.J.vv);

    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  1, sizeof(cl_mem),    (void*)&ocl->dev.U1);
    
    ocl->err = clSetKernelArg(ocl->vtx_bnd2,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_bnd2,  1, sizeof(cl_float8), (void*)&msh->mat_prm);
    ocl->err = clSetKernelArg(ocl->vtx_bnd2,  2, sizeof(cl_mem),    (void*)&ocl->dev.U1);
    
    ocl->err = clSetKernelArg(ocl->vtx_bnd3,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_bnd3,  1, sizeof(cl_float8), (void*)&msh->mat_prm);
    ocl->err = clSetKernelArg(ocl->vtx_bnd3,  2, sizeof(cl_mem),    (void*)&ocl->dev.F1);
    ocl->err = clSetKernelArg(ocl->vtx_bnd3,  3, sizeof(cl_mem),    (void*)&ocl->dev.J.vv);
    
    ocl->err = clSetKernelArg(ocl->vtx_bnd4,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_bnd4,  1, sizeof(cl_mem),    (void*)&ocl->dev.F1);
    ocl->err = clSetKernelArg(ocl->vtx_bnd4,  2, sizeof(cl_mem),    (void*)&ocl->dev.J.vv);
}


//final
void ocl_final(struct msh_obj *msh, struct ocl_obj *ocl)
{
    ocl->err = clFlush(ocl->command_queue);
    ocl->err = clFinish(ocl->command_queue);
    
    //kernels
    ocl->err = clReleaseKernel(ocl->vtx_init);
    ocl->err = clReleaseKernel(ocl->vtx_assm);
    ocl->err = clReleaseKernel(ocl->vtx_bnd1);
    ocl->err = clReleaseKernel(ocl->vtx_bnd2);
    ocl->err = clReleaseKernel(ocl->vtx_bnd3);
    ocl->err = clReleaseKernel(ocl->vtx_bnd4);
    
    //device
    ocl->err = clReleaseMemObject(ocl->dev.vtx_xx);
    
    ocl->err = clReleaseMemObject(ocl->dev.U0);
    ocl->err = clReleaseMemObject(ocl->dev.U1);
    ocl->err = clReleaseMemObject(ocl->dev.F1);
    
    ocl->err = clReleaseMemObject(ocl->dev.J.ii);
    ocl->err = clReleaseMemObject(ocl->dev.J.jj);
    ocl->err = clReleaseMemObject(ocl->dev.J.vv);
    
    ocl->err = clReleaseProgram(ocl->program);
    ocl->err = clReleaseCommandQueue(ocl->command_queue);
    ocl->err = clReleaseContext(ocl->context);
    
    //host
    free(ocl->hst.vtx_xx);
    
    free(ocl->hst.U0);
    free(ocl->hst.U1);
    free(ocl->hst.F1);
    
    free(ocl->hst.J.ii);
    free(ocl->hst.J.jj);
    free(ocl->hst.J.vv);
    
    return;
}



#endif /* ocl_h */
