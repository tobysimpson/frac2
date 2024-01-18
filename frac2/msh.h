//
//  msh.h
//  frac2
//
//  Created by Toby Simpson on 12.01.24.
//

#ifndef msh_h
#define msh_h

//object
struct msh_obj
{
    cl_int3     ele_dim;
    cl_int3     vtx_dim;
    
    cl_float3   x0;
    cl_float3   x1;
    cl_float3   dx;
    
    cl_float8   mat_prm;
    
    int         ne_tot;     //totals
    int         nv_tot;
};

//init
void msh_init(struct msh_obj *msh)
{
    //dim
    msh->ele_dim.x = 2;
    msh->ele_dim.y = msh->ele_dim.x;
    msh->ele_dim.z = msh->ele_dim.x;
    
    msh->vtx_dim = (cl_int3){msh->ele_dim.x+1, msh->ele_dim.y+1, msh->ele_dim.z+1};
    
    printf("ele_dim %d %d %d\n", msh->ele_dim.x, msh->ele_dim.y, msh->ele_dim.z);
    printf("vtx_dim %d %d %d\n", msh->vtx_dim.x, msh->vtx_dim.y, msh->vtx_dim.z);
    
    //range
    msh->x0 = (cl_float3){+0e+0f,+0e+0f,+0e+0f};
    msh->x1 = (cl_float3){+1e+0f,+1e+0f,+1e+0f};
//    msh->x1 = (cl_float3){msh->ele_dim.x, msh->ele_dim.y, msh->ele_dim.z};
    msh->dx = (cl_float3){(msh->x1.x - msh->x0.x)/(float)msh->ele_dim.x, (msh->x1.y - msh->x0.y)/(float)msh->ele_dim.y, (msh->x1.z - msh->x0.z)/(float)msh->ele_dim.z};
    
    printf("x0 %+e %+e %+e\n", msh->x0.x, msh->x0.y, msh->x0.z);
    printf("x1 %+e %+e %+e\n", msh->x1.x, msh->x1.y, msh->x1.z);
    printf("dx %+e %+e %+e\n", msh->dx.x, msh->dx.y, msh->dx.z);
    
    //material params
    msh->mat_prm.s0 = 121.15f;                                  //lamé      lambda
    msh->mat_prm.s1 = 80.77f;                                   //lamé      mu
    msh->mat_prm.s2 = 2.7e-3f;                                  //constant  Gc  = energy release
    msh->mat_prm.s3 = 2e0f*msh->dx.x;                           //constant  ls or eps = length scale
    msh->mat_prm.s4 = msh->mat_prm.s2/msh->mat_prm.s3;          //Gc/ls     (pre-calc)
    msh->mat_prm.s5 = msh->mat_prm.s2*msh->mat_prm.s3;          //Gc*ls     (pre-calc)
    msh->mat_prm.s6 = msh->mat_prm.s4*(1e+4f - 1e0f);           //gamma     tau_irr = 1e-2 (kopa2023 eq7) gamma = (gc/ls)*(1/tau^2 - 1)
    msh->mat_prm.s7 = 0e0f;                                     //displacement bc

    
//    printf("mat_prm %e %e %e %e\n", msh->mat_prm.s0, msh->mat_prm.s1, msh->mat_prm.z, msh->mat_prm.w);
    printf("mat_prm.s0 %f\n", msh->mat_prm.s0);
    printf("mat_prm.s1 %f\n", msh->mat_prm.s1);
    printf("mat_prm.s2 %f\n", msh->mat_prm.s2);
    printf("mat_prm.s3 %f\n", msh->mat_prm.s3);
    printf("mat_prm.s4 %f\n", msh->mat_prm.s4);
    printf("mat_prm.s5 %f\n", msh->mat_prm.s5);
    printf("mat_prm.s6 %e\n", msh->mat_prm.s6);
    printf("mat_prm.s7 %e\n", msh->mat_prm.s7);
    
    //totals
    msh->ne_tot = msh->ele_dim.x*msh->ele_dim.y*msh->ele_dim.z;
    msh->nv_tot = msh->vtx_dim.x*msh->vtx_dim.y*msh->vtx_dim.z;
    
    printf("ne_tot=%d\n", msh->ne_tot);
    printf("nv_tot=%d\n", msh->nv_tot);
    
    return;
}


#endif /* msh_h */
