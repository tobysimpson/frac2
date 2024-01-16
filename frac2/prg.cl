//
//  prg.cl
//  frac2
//
//  Created by Toby Simpson on 12.01.24.
//


/*
 ===================================
 prototypes
 ===================================
 */

int     fn_idx1(int3 pos, int3 dim);
int     fn_idx3(int3 pos);

int     fn_bnd1(int3 pos, int3 dim);
int     fn_bnd2(int3 pos, int3 dim);

void    bas_eval(float3 p, float ee[8]);
void    bas_grad(float3 p, float3 gg[8], float3 dx);
float4  bas_itpe(float4 uu2[8], float bas_ee[8]);
void    bas_itpg(float4 uu2[8], float3 bas_gg[8], float3 u_grad[3]);

float3  mtx_mv(float3 A[3], float3 v);
void    mtx_mm(float3 A[3], float3 B[3], float3 C[3]);
void    mtx_mmT(float3 A[3], float3 B[3], float3 C[3]);
void    mtx_mdmT(float3 A[3], float D[3], float3 B[3], float3 C[3]);
void    mtx_sum(float3 A[3], float3 B[3], float3 C[3]);

float   sym_tr(float8 A);
float   sym_det(float8 A);
float8  sym_vvT(float3 v);
float3  sym_mv(float8 A, float3 v);
float8  sym_mm(float8 A, float8 B);
float8  sym_mdmT(float3 A[3], float D[3]);
float8  sym_sumT(float3 A[3]);
float   sym_tip(float8 A, float8 B);

float8  mec_e(float3 g[3]);
float8  mec_s(float8 E, float8 mat_prm);
float   mec_p(float8 E, float8 mat_prm);
float   mec_p1(float D[3], float8 mat_prm);
//void    mec_e12(float D[3], float3 V[3], float8 E1, float8 E2);
//void    mec_s12(float D[3], float3 V[3], float8 mat_prm, float8 S1, float8 S2);
float8  mec_test(float D[3], float3 V[3]);

void    mem_gr3(global float4 *buf, float4 uu3[27], int3 pos, int3 dim);
void    mem_lr2(float4 uu3[27], float4 uu2[8], int3 pos);

void    eig_val(float8 A, float dd[3]);
void    eig_vec(float8 A, float dd[3], float3 vv[3]);
void    eig_dcm(float8 A, float dd[3], float3 vv[3]);
void    eig_drv(float8 dA, float D[3], float3 V[3], float8 A1, float8 A2);
float   eig_dpdu(float D[3], float3 V[3], float3 dU[3], float8 mat_prm);


/*
 ===================================
 constants
 ===================================
 */

constant int3 off2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};

constant int3 off3[27] = {
    {0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},
    {0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},
    {0,0,2},{1,0,2},{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}};


/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx1(int3 pos, int3 dim)
{
    return pos.x + dim.x*(pos.y + dim.y*pos.z);
}

//index 3x3x3
int fn_idx3(int3 pos)
{
    return pos.x + 3*pos.y + 9*pos.z;
}

//in-bounds
int fn_bnd1(int3 pos, int3 dim)
{
    return all(pos>=0)*all(pos<dim);
}

//on the boundary
int fn_bnd2(int3 pos, int3 dim)
{
    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);
}

/*
 ===================================
 quadrature [0,1]
 ===================================
 */

//1-point gauss [0,1]
constant float qp1 = 5e-1f;
constant float qw1 = 1e+0f;

//2-point gauss [0,1]
constant float qp2[2] = {0.211324865405187f,0.788675134594813f};
constant float qw2[2] = {5e-1f,5e-1f};

//3-point gauss [0,1]
constant float qp3[3] = {0.112701665379258f,0.500000000000000f,0.887298334620742f};
constant float qw3[3] = {0.277777777777778f,0.444444444444444f,0.277777777777778f};

/*
 ===================================
 basis
 ===================================
 */

//eval at qp
void bas_eval(float3 p, float ee[8])
{
    float x0 = 1e0f - p.x;
    float y0 = 1e0f - p.y;
    float z0 = 1e0f - p.z;
    
    float x1 = p.x;
    float y1 = p.y;
    float z1 = p.z;
    
    ee[0] = x0*y0*z0;
    ee[1] = x1*y0*z0;
    ee[2] = x0*y1*z0;
    ee[3] = x1*y1*z0;
    ee[4] = x0*y0*z1;
    ee[5] = x1*y0*z1;
    ee[6] = x0*y1*z1;
    ee[7] = x1*y1*z1;
    
    return;
}

//grad at qp
void bas_grad(float3 p, float3 gg[8], float3 dx)
{
    float x0 = 1e0f - p.x;
    float y0 = 1e0f - p.y;
    float z0 = 1e0f - p.z;
    
    float x1 = p.x;
    float y1 = p.y;
    float z1 = p.z;
    
    //{/dx,/dy,/dz}
    gg[0] = (float3){-y0*z0, -x0*z0, -x0*y0}/dx;
    gg[1] = (float3){+y0*z0, -x1*z0, -x1*y0}/dx;
    gg[2] = (float3){-y1*z0, +x0*z0, -x0*y1}/dx;
    gg[3] = (float3){+y1*z0, +x1*z0, -x1*y1}/dx;
    gg[4] = (float3){-y0*z1, -x0*z1, +x0*y0}/dx;
    gg[5] = (float3){+y0*z1, -x1*z1, +x1*y0}/dx;
    gg[6] = (float3){-y1*z1, +x0*z1, +x0*y1}/dx;
    gg[7] = (float3){+y1*z1, +x1*z1, +x1*y1}/dx;
    
    return;
}

//interp eval
float4 bas_itpe(float4 uu2[8], float bas_ee[8])
{
    float4 u = 0e0f;
    
    for(int i=0; i<8; i++)
    {
        u += uu2[i]*bas_ee[i];
    }
    return u;
}


//interp grad, u_grad[i] = du[i]/{dx,dy,dz}, rows of Jacobian
void bas_itpg(float4 uu2[8], float3 bas_gg[8], float3 u_grad[4])
{
    for(int i=0; i<8; i++)
    {
        u_grad[0] += uu2[i].x*bas_gg[i];
        u_grad[1] += uu2[i].y*bas_gg[i];
        u_grad[2] += uu2[i].z*bas_gg[i];
        u_grad[3] += uu2[i].w*bas_gg[i];
    }
    return;
}


/*
 ===================================
 memory
 ===================================
 */


//global read 3x3x3 vectors
void mem_gr3(global float4 *buf, float4 uu3[27], int3 pos, int3 dim)
{
    for(int i=0; i<27; i++)
    {
        int3 adj_pos1 = pos + off3[i] - 1;
        int  adj_idx1 = fn_idx1(adj_pos1, dim);

        //copy/cast
        uu3[i] = buf[adj_idx1];
    }
    return;
}

//local read 2x2x2 from 3x3x3 vector
void mem_lr2(float4 uu3[27], float4 uu2[8], int3 pos)
{
    for(int i=0; i<8; i++)
    {
        int3 adj_pos3 = pos + off2[i];
        int  adj_idx3 = fn_idx3(adj_pos3);

        //copy
        uu2[i] = uu3[adj_idx3];
    }
    return;
}


/*
 ===================================
 matrix R^3x3
 ===================================
 */

//mmult Av
float3 mtx_mv(float3 A[3], float3 v)
{
    return A[0]*v.x + A[1]*v.y + A[2]*v.z;
}

//mmult C = AB
void mtx_mm(float3 A[3], float3 B[3], float3 C[3])
{
    C[0] = mtx_mv(A,B[0]);
    C[1] = mtx_mv(A,B[1]);
    C[2] = mtx_mv(A,B[2]);

    return;
}

//mmult C = AB^T
void mtx_mmT(float3 A[3], float3 B[3], float3 C[3])
{
    C[0] = A[0]*B[0].x + A[1]*B[1].x + A[2]*B[2].x;
    C[1] = A[0]*B[0].y + A[1]*B[1].y + A[2]*B[2].y;
    C[2] = A[0]*B[0].z + A[1]*B[1].z + A[2]*B[2].z;

    return;
}

//mmult C = ADB^T, diagonal D
void mtx_mdmT(float3 A[3], float D[3], float3 B[3], float3 C[3])
{
    C[0] = D[0]*A[0]*B[0].x + D[1]*A[1]*B[1].x + D[2]*A[2]*B[2].x;
    C[1] = D[0]*A[0]*B[0].y + D[1]*A[1]*B[1].y + D[2]*A[2]*B[2].y;
    C[2] = D[0]*A[0]*B[0].z + D[1]*A[1]*B[1].z + D[2]*A[2]*B[2].z;

    return;
}

//sum
void mtx_sum(float3 A[3], float3 B[3], float3 C[3])
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + B[2];
}

/*
 ===================================
 symmetric R^3x3
 ===================================
 */

//sym trace
float sym_tr(float8 A)
{
    return A.s0 + A.s3 + A.s5;
}

//sym determinant
float sym_det(float8 A)
{
    return dot((float3){A.s0,A.s1,A.s2}, cross((float3){A.s1, A.s3, A.s4}, (float3){A.s2, A.s4, A.s5}));
//    return A.s0*A.s3*A.s5 - (A.s0*A.s4*A.s4 + A.s2*A.s2*A.s3 + A.s1*A.s1*A.s5) + 2e0f*A.s1*A.s2*A.s4;
}

//outer product vv^T
float8 sym_vvT(float3 v)
{
    return (float8){v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z, 0e0f, 0e0f};
}

//sym mtx-vec
float3 sym_mv(float8 A, float3 v)
{
    return (float3){dot(A.s012,v), dot(A.s134,v), dot(A.s245,v)};
}

//sym mtx-mtx
float8 sym_mm(float8 A, float8 B)
{
    return (float8){A.s0*B.s0 + A.s1*B.s1 + A.s2*B.s2,
                    A.s0*B.s1 + A.s1*B.s3 + A.s2*B.s4,
                    A.s0*B.s2 + A.s1*B.s4 + A.s2*B.s5,
                    A.s1*B.s1 + A.s3*B.s3 + A.s4*B.s4,
                    A.s1*B.s2 + A.s3*B.s4 + A.s4*B.s5,
                    A.s2*B.s2 + A.s4*B.s4 + A.s5*B.s5, 0e0f, 0e0f};
}

//mul A = VDV^T, diagonal D
float8  sym_mdmT(float3 V[3], float D[3])
{
    float8 A;
    
    A.s0 = D[0]*V[0].x*V[0].x + D[1]*V[1].x*V[1].x + D[2]*V[2].x*V[2].x;
    A.s1 = D[0]*V[0].x*V[0].y + D[1]*V[1].x*V[1].y + D[2]*V[2].x*V[2].y;
    A.s2 = D[0]*V[0].x*V[0].z + D[1]*V[1].x*V[1].z + D[2]*V[2].x*V[2].z;
    A.s3 = D[0]*V[0].y*V[0].y + D[1]*V[1].y*V[1].y + D[2]*V[2].y*V[2].y;
    A.s4 = D[0]*V[0].y*V[0].z + D[1]*V[1].y*V[1].z + D[2]*V[2].y*V[2].z;
    A.s5 = D[0]*V[0].z*V[0].z + D[1]*V[1].z*V[1].z + D[2]*V[2].z*V[2].z;
    A.s6 = 0e0f;
    A.s7 = 0e0f;
    
    return A;
}

//sum S = A+A^T
float8 sym_sumT(float3 A[3])
{
    float8 S;
    
    S.s0 = A[0].x + A[0].x;
    S.s1 = A[1].x + A[0].y;
    S.s2 = A[2].x + A[0].z;
    S.s3 = A[1].y + A[1].y;
    S.s4 = A[2].y + A[1].z;
    S.s5 = A[2].z + A[2].z;
    
    return S;
}

//sym tensor inner prod
float sym_tip(float8 A, float8 B)
{
    return A.s0*B.s0 + 2e0f*A.s1*B.s1 + 2e0f*A.s2*B.s2 + A.s3*B.s3 + 2e0f*A.s4*B.s4 + A.s5*B.s5;
}

/*
 ===================================
 mechanics
 ===================================
 */

//strain (du + du^T)/2
float8 mec_e(float3 du[3])
{
    return 5e-1f*sym_sumT(du);
}

//stress pk2 = lam*tr(e)*I + 2*mu*e
float8 mec_s(float8 E, float8 mat_prm)
{
    float8 S = 2e0f*mat_prm.s1*E;
    S.s035 += mat_prm.s0*sym_tr(E);
    
    return S;
}

//energy phi = 0.5*lam*(tr(E))^2 + mu*tr(E^2)
float mec_p(float8 E, float8 mat_prm)
{
    return 5e-1f*mat_prm.s0*pown(sym_tr(E),2) + mat_prm.s1*sym_tr(sym_mm(E,E));
}


//energy pos (miehe2010)
float mec_p1(float D[3], float8 mat_prm)
{
    float s = D[0] + D[1] + D[2];
    
    float d0 = (D[0]>0e0f)*D[0];
    float d1 = (D[1]>0e0f)*D[1];
    float d2 = (D[2]>0e0f)*D[2];
    
    return 5e-1f*mat_prm.s0*s*s*(s>0e0f) + mat_prm.s1*(d0*d0 + d1*d1 + d2*d2);
}

////strain split
//void mec_e12(float D[3], float3 V[3], float8 E1, float8 E2)
//{
//    float8 vv0 = sym_vvT(V[0]);
//    float8 vv1 = sym_vvT(V[1]);
//    float8 vv2 = sym_vvT(V[2]);
//    
//    //strain
//    E1 = (D[0]>0e0f)*D[0]*vv0 + (D[1]>0e0f)*D[1]*vv1 + (D[2]>0e0f)*D[2]*vv2;
//    E2 = (D[0]<0e0f)*D[0]*vv0 + (D[1]<0e0f)*D[1]*vv1 + (D[2]<0e0f)*D[2]*vv2;
//    
//    return;
//}

//stress split Miehe2010
float8  mec_test(float D[3], float3 V[3])
{
    float8 S1 = 0e0f;
    
    for(int i=0; i<3; i++)
    {
        S1 += sym_vvT(V[i]);
    }
    
    return S1;
}


/*
 ===================================
 eigs (sym 3x3)
 ===================================
 */

//eigenvalues (A real symm) - Deledalle2017
void eig_val(float8 A, float D[3])
{
    //weird layout
    float a = A.s0;
    float b = A.s3;
    float c = A.s5;
    float d = A.s1;
    float e = A.s4;
    float f = A.s2;
    
    float x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3e0f*(d*d + e*e + f*f);
    float x2 = -(2e0f*a - b - c)*(2e0f*b - a - c)*(2e0f*c - a - b) + 9e0f*(2e0f*c - a - b)*d*d + (2e0f*b - a - c)*f*f + (2e0f*a - b - c)*e*e - 5.4e1f*d*e*f;
    
    float p1 = atan(sqrt(4e0f*x1*x1*x1 - x2*x2)/x2);
    
    //logic
    float phi = 5e-1f*M_PI_F;
    phi = (x2>0e0f)?p1         :phi;       //x2>0
    phi = (x2<0e0f)?p1 + M_PI_F:phi;       //x2<0
 
    //write
    D[0] = (a + b + c - 2e0f*sqrt(x1)*cos((phi         )/3e0f))/3e0f;
    D[1] = (a + b + c + 2e0f*sqrt(x1)*cos((phi - M_PI_F)/3e0f))/3e0f;
    D[2] = (a + b + c + 2e0f*sqrt(x1)*cos((phi + M_PI_F)/3e0f))/3e0f;
    
    return;
}


//eigenvectors (A real symm) - Kopp2008
void eig_vec(float8 A, float D[3], float3 V[3])
{
    //cross, normalise, skip when lam=0
    V[0] = normalize(cross((float3){A.s0-D[0], A.s1, A.s2},(float3){A.s1, A.s3-D[0], A.s4}))*(D[0]!=0e0f);
    V[1] = normalize(cross((float3){A.s0-D[1], A.s1, A.s2},(float3){A.s1, A.s3-D[1], A.s4}))*(D[1]!=0e0f);
    V[2] = normalize(cross((float3){A.s0-D[2], A.s1, A.s2},(float3){A.s1, A.s3-D[2], A.s4}))*(D[2]!=0e0f);

    return;
}


//eigen decomposition
void eig_dcm(float8 A, float D[3], float3 V[3])
{
    eig_val(A, D);
    eig_vec(A, D, V);
    
    return;
}

//derivative of A in direction of dA where A = VDV^T, Jodlbauer2020, dA arrives transposed
void eig_drv(float8 dA, float D[3], float3 V[3], float8 S1, float8 S2)
{
    //L = (A - D[i]*I)
    
    //derivs, per eig
    float  dD[3];
    float3 dV[3];
    
    //split (1=pos, 2=neg)
    float dD1[3];
    float dD2[3];
    float D1[3];
    float D2[3];
    
    //loop eigs
    for(int i=0; i<3; i++)
    {
        //Dinv inverse
        float  Dinv[3];
        Dinv[0] = (D[0]==D[i])?0e0f:1e0f/(D[0]-D[i]);
        Dinv[1] = (D[1]==D[i])?0e0f:1e0f/(D[1]-D[i]);
        Dinv[2] = (D[2]==D[i])?0e0f:1e0f/(D[2]-D[i]);
        
        //L inverse
        float8 Linv = sym_mdmT(V,Dinv);
        
        //derivs
        dV[i] = -sym_mv(sym_mm(Linv, dA), V[i]);
        dD[i] = dot(V[i], sym_mv(dA, V[i]));
        
        //split
        dD1[i] = (D[i]>0e0f)?dD[i]:0e0f;
        dD2[i] = (D[i]<0e0f)?dD[i]:0e0f;
        
        D1[i] = (D[i]>0e0f)?D[i]:0e0f;
        D2[i] = (D[i]<0e0f)?D[i]:0e0f;
        
    }//i
    
    //A_pos = VD_posV^T
    
    float3 M1[3];               //dV*D_pos*V^T
    float3 M2[3];               //dV*D_neg*V^T
    mtx_mdmT(dV,D1,V,M1);
    mtx_mdmT(dV,D2,V,M2);
    
    //finally, S1/S2, pos/neg
    S1 = sym_sumT(M1) + sym_mdmT(V,dD1);
    S2 = sym_sumT(M2) + sym_mdmT(V,dD2);
    
    return;
}

//derivative of energy wrt disp  d(energy+)/du
float eig_dpdu(float D[3], float3 V[3], float3 dU[3], float8 mat_prm)
{
    //derivatives of principal strains wrt perturbation
    float dD[3] = {dot(V[0], mtx_mv(dU,V[0])), dot(V[1], mtx_mv(dU,V[1])), dot(V[2], mtx_mv(dU,V[2]))};
    
    float trE = (D[0]+D[1]+D[2]);
    
    //test pos trace for first part, inidividual eigs (primary strains) for second part
    return mat_prm.s0*(trE>0e0f)*(dD[0] + dD[1] + dD[2]) + 2e0f*mat_prm.s1*((D[0]>0e0f)*dD[0] + (D[1]>0e0f)*dD[1] + (D[2]>0e0f)*dD[2]);
}


/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_init(const  int3    vtx_dim,
                     const  float3  x0,
                     const  float3  dx,
                     global float4  *vtx_xx,
                     global float4  *U0,
                     global float4  *U1,
                     global float4  *F1,
                     global int16   *J_ii,
                     global int16   *J_jj,
                     global float16 *J_vv)
{
    int3 vtx1_pos1 = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
//    printf("vtx_dim %v3d\n", vtx_dim);

    //coord
    vtx_xx[vtx1_idx1].xyz = x0 + dx*convert_float3(vtx1_pos1);

//    printf("xx %2d %v3f\n", vtx1_idx1, vtx_xx[vtx1_idx1]);
    
    //rhs
    U0[vtx1_idx1] = 0e0f;
    U1[vtx1_idx1] = 0e0f;
    F1[vtx1_idx1] = 0e0f;
    
    //vtx2
    for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
    {
        int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
        int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);
        int  vtx2_bnd1 = fn_bnd1(vtx2_pos1, vtx_dim);
        
        //block
        int idx1 = 27*vtx1_idx1 + vtx2_idx3;
        
        //block pointers
        global int*   ii = (global int*)&J_ii[idx1];
        global int*   jj = (global int*)&J_jj[idx1];
        global float* vv = (global float*)&J_vv[idx1];
  
        //dim1
        for(int dim1=0; dim1<4; dim1++)
        {
            //dim2
            for(int dim2=0; dim2<4; dim2++)
            {
                //uu
                int idx2 = 4*dim1 + dim2;
                ii[idx2] = vtx2_bnd1*(4*vtx1_idx1 + dim1);
                jj[idx2] = vtx2_bnd1*(4*vtx2_idx1 + dim2);
                vv[idx2] = 0e0f; //vtx2_bnd1*(4*dim1 + dim2 + 1);
                
            } //dim2
            
        } //dim1
        
    } //vtx2
    
    return;
}


//assemble
kernel void vtx_assm(const  int3     vtx_dim,
                     const  float3   dx,
                     const  float8   mat_prm,
                     global float4   *U0,
                     global float4   *U1,
                     global float4   *F1,
                     global float16  *J_vv)
{
    int3 ele_dim = vtx_dim - 1;
    int3 vtx1_pos1  = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //volume
    float vlm = dx.x*dx.y*dx.z;
    
    printf("vtx1 %3d %v3d %e\n", vtx1_idx1, vtx1_pos1, vlm);
    
    //read 3x3x3
    float4  uu30[27]; //prev
    float4  uu31[27];
    mem_gr3(U0, uu30, vtx1_pos1, vtx_dim);
    mem_gr3(U1, uu31, vtx1_pos1, vtx_dim);


    //ref - avoids -ve int bug
    int  vtx1_idx2 = 8;
    
    //ele1
    for(uint ele1_idx2=0; ele1_idx2<8; ele1_idx2++)
    {
        int3 ele1_pos2 = off2[ele1_idx2];
        int3 ele1_pos1 = vtx1_pos1 + ele1_pos2 - 1;
        int  ele1_bnd1 = fn_bnd1(ele1_pos1, ele_dim);
        
        //ref vtx (decrement to avoid bug)
        vtx1_idx2 -= 1;
        
        //in-bounds
        if(ele1_bnd1)
        {
            //read 2x2x2
            float4 uu20[8];  //prev
            float4 uu21[8];
            mem_lr2(uu30, uu20, ele1_pos2);
            mem_lr2(uu31, uu21, ele1_pos2);
            
            //qpt1 (change limit with scheme 1,8,27)
            for(int qpt1=0; qpt1<1; qpt1++)
            {
                //1pt
                float3 qp = (float3){qp1,qp1,qp1};
                float  qw = qw1*qw1*qw1*vlm;
                
//                //2pt
//                float3 qp = (float3){qp2[off2[qpt1].x], qp2[off2[qpt1].y], qp2[off2[qpt1].z]};
//                float  qw = qw2[off2[qpt1].x]*qw2[off2[qpt1].y]*qw2[off2[qpt1].z]*vlm;
                
//                //3pt
//                float3 qp = (float3){qp3[off3[qpt1].x], qp3[off3[qpt1].y], qp3[off3[qpt1].z]};
//                float  qw = qw3[off3[qpt1].x]*qw3[off3[qpt1].y]*qw3[off3[qpt1].z]*vlm;
                
                //basis
                float  bas_ee[8];
                float3 bas_gg[8];
                bas_eval(qp, bas_ee);
                bas_grad(qp, bas_gg, dx);
                
                //itp val
                float4 itp_u0 = bas_itpe(uu20, bas_ee);
                float4 itp_u1 = bas_itpe(uu21, bas_ee);
                
                //itp grad
                float3 du[4] = {{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f}};
                bas_itpg(uu21, bas_gg, du);
                
//                printf("%v3e\n", du[0]);
//                printf("%v3e\n", du[1]);
//                printf("%v3e\n", du[2]);
//                printf("\n");
                
                //itp crack
                float c0 = itp_u0.w;
                float c1 = itp_u1.w;
                float dc = c1 - c0;              //for heaviside
                float cc[2] = {pown(1e0f - c1, 2), 2e0f*(c1 - 1e0f)};  //pre-calc
            
                //strain
                float8 E = mec_e(du);
                float trE = sym_tr(E);
                
//                printf("%v8e\n", E); //ok
                
                //decompose E
                float  D[3];
                float3 V[3];
                eig_dcm(E, D, V);
                
//                printf("%e %e %e\n", D[0],D[1],D[2]); //ok
//                printf("%v3e %v3e %v3e\n", V[0],V[1],V[2]); //ok
                
                //energy (pos)
                float p1 = mec_p1(D, mat_prm);
                
                //split stress
                float8 S1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                float8 S2 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                
        
                //doesnt like function do in place
                for(int i=0; i<3; i++)
                {
                    float8 vvT = sym_vvT(V[i]);
                    
                    S1 += mat_prm.s0*trE*(trE>0e0f) + 2e0f*mat_prm.s1*D[i]*(D[i]>0e0f)*vvT;
                    S2 += mat_prm.s0*trE*(trE<0e0f) + 2e0f*mat_prm.s1*D[i]*(D[i]<0e0f)*vvT;
                }
                
//                printf("%v8f\n",S1);
//                printf("%v8f\n",S2);

                //local ptr
                global float* ff = (global float*)&F1[vtx1_idx1];
                
                //rhs u
                for(int dim1=0; dim1<3; dim1++)
                {
                    //tensor basis
                    float3 du1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                    du1[dim1] = bas_gg[vtx1_idx2];

                    //strain
                    float8 E1 = mec_e(du1);
                    
//                    printf("%+v8f\n", E1); //ok
                    
                    //write
                    ff[dim1] += sym_tip(cc[0]*S1 + S2, E1)*qw;
                }
                
                //rhs c
                ff[3] += ((cc[1]*p1 + mat_prm.s4*c1 + mat_prm.s6*(dc)*(dc<0e0f))*bas_ee[vtx1_idx2] + mat_prm.s5*dot(du[3], bas_gg[vtx1_idx2]))*qw;
               
                //vtx2
                for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
                {
                    //idx
                    int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
                    int  vtx2_idx3 = fn_idx3(vtx2_pos3);
                    
                    //idx, local ptr
                    int idx1 = 27*vtx1_idx1 + vtx2_idx3;
                    global float* vv = (global float*)&J_vv[idx1];
                    
                    //cc write
                    vv[15] += qw; //((2e0f*p1 + mat_prm.s6 + mat_prm.s7*(dc<0e0f))*bas_ee[vtx1_idx2]*bas_ee[vtx2_idx2] + mat_prm.s4*mat_prm.s5*dot(bas_gg[vtx1_idx2], bas_gg[vtx2_idx2]))*qw;
                    
                    //dim1
                    for(int dim1=0; dim1<3; dim1++)
                    {
                        //tensor basis
                        float3 du1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                        du1[dim1] = bas_gg[vtx1_idx2];

                        //strain
                        float8 E1 = mec_e(du1);
                        
                        //idx
                        int idx3 = 4*dim1 + 3;
                        int idx4 = 12 + dim1;
                        
                        //coupling uc/cu
                        float cpl = cc[1]*eig_dpdu(D, V, du1, mat_prm)*bas_ee[vtx1_idx2]*qw;
                        
                        //uc write
                        vv[idx3] += cpl;
                        vv[idx4] += cpl;
                        
                        //dim2
                        for(int dim2=0; dim2<3; dim2++)
                        {
                            //tensor basis
                            float3 du2[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                            du2[dim2] = bas_gg[vtx2_idx2];
                            
                            //strain
                            float8 E2 = mec_e(du2);
                            float trE2 = sym_tr(E2);
                            
                            //split stress (deriv)
                            float8 dS1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                            float8 dS2 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                            
                            //split (strain)
                            eig_drv(E2, D, V, dS1, dS2);
                            
                            //stress (lam*tr(E)I + 2*mu*E, pos/neg)
                            dS1 = 2e0f*mat_prm.s1*dS1;
                            dS1.s035 += mat_prm.s0*(trE>0e0f)*(trE2);
                            
                            dS2 = 2e0f*mat_prm.s1*dS2;
                            dS2.s035 += mat_prm.s0*(trE<0e0f)*(trE2);
                            
                            //write uu
                            int idx2 = 4*dim1 + dim2;
                            vv[idx2] += (cc[0]*sym_tip(dS1, E1) + sym_tip(dS2, E1))*qw;
                            
                        } //dim2
                        
                    } //dim1
                    
                } //vtx2
                 
            } //qpt
            
        } //ele1_bnd1
        
    } //ele
    
    return;
}


//notch
kernel void vtx_bnd1(const  int3   vtx_dim,
                     global float4 *U1)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //bools
    int b1 = (vtx1_pos1.z == ((vtx_dim.z - 1)/2));  //halfway up
    int b2 = (vtx1_pos1.x <= ((vtx_dim.x - 1)/2));  //halfway across
    
    //notch
    if(b1&&b2)
    {
        U1[vtx1_idx1].w = 1e0f;
    }

    return;
}

//displacement
kernel void vtx_bnd2(const  int3   vtx_dim,
                     const  float8 mat_prm,
                     global float4 *U1)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //bools
    int b1 = (vtx1_pos1.z == 0);                    //base
    int b2 = (vtx1_pos1.z == (vtx_dim.z - 1));      //top
    
    //base
    if(b1)
    {
        //soln
        U1[vtx1_idx1].xyz = (float3){0e0f,0e0f,0e0f};
    }
    
    //top
    if(b2)
    {
        //soln
        U1[vtx1_idx1].xyz = (float3){0e0f,0e0f,mat_prm.s7};
    }

    return;
}


//dirichlet
kernel void vtx_bnd3(const  int3    vtx_dim,
                     const  float8  mat_prm,
                     global float4  *F1,
                     global float16 *J_vv)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //bools
    int b1 = (vtx1_pos1.z == 0);                    //base
    int b2 = (vtx1_pos1.z == (vtx_dim.z - 1));      //top
    
    //base
    if(b1)
    {
        //rhs
        F1[vtx1_idx1].xyz = (float3){0e0f,0e0f,0e0f};
    }
    
    //top
    if(b2)
    {
        //rhs
        F1[vtx1_idx1].xyz = (float3){0e0f,0e0f,mat_prm.s7};
    }

    //I
    if(b1+b2) //or
    {
        //vtx2
        for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
        {
            int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
            int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);
            int  vtx2_bnd1 = fn_bnd1(vtx2_pos1, vtx_dim);
            
            //local pointer
            int idx1 = 27*vtx1_idx1 + vtx2_idx3;
            global float* vv = (global float*)&J_vv[idx1];
            

            //dim1
            for(int dim1=0; dim1<4; dim1++)
            {
                //dim2
                for(int dim2=0; dim2<4; dim2++)
                {
                    //uu
                    int idx2 = 4*dim1 + dim2;
                    vv[idx2] = vtx2_bnd1*(vtx1_idx1==vtx2_idx1)*(dim1==dim2);
                    
                } //dim2
                
            } //dim1
            
        } //vtx2
        
    } //if
    
    return;
}

