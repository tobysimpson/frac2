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
float16 bas_itpg(float4 uu2[8], float3 bas_gg[8]);
float16 bas_tens(int dim, float3 g);

float3  mtx_mv(float16 A, float3 v);
float16 mtx_mm(float16 A, float16 B);
float16 mtx_uvT(float3 u, float3 v);
float16 mtx_mT(float16 A);
float16 mtx_md(float16 A, float3 D);

float   sym_tr(float8 A);
float   sym_det(float8 A);
float8  sym_vvT(float3 v);
float3  sym_mv(float8 A, float3 v);
float8  sym_mm(float8 A, float8 B);
float8  sym_mdmT(float16 V, float3 D);
float8  sym_sumT(float16 A);
float   sym_tip(float8 A, float8 B);

float8  mec_E(float16 du);
float8  mec_S(float8 E, float8 mat_prm);
float   mec_p(float8 E, float8 mat_prm);
float   mec_p1(float3 D, float8 mat_prm);

void    mec_D1D2(float3 D, float3 *D1, float3 *D2);
void    mec_S1S2(float3 D, float16 V, float8 mat_prm, float8 *S1, float8 *S2);
void    mec_dS1dS2(float8 dA, float3 D, float16 V, float8 *dS1, float8 *dS2);
float   mec_dp1(float3 D, float16 V, float16 dU, float8 mat_prm);

void    mem_gr3(global float4 *buf, float4 uu3[27], int3 pos, int3 dim);
void    mem_lr2(float4 uu3[27], float4 uu2[8], int3 pos);

float3  eig_val(float8 A);
float16 eig_vec(float8 A, float3 D);
float8  eig_mpinv(float lam, float3 D, float16 V);






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


//interp grad, Jacobian, u_grad[i] = du[i](rows)/{dx,dy,dz}(cols)
float16 bas_itpg(float4 uu2[8], float3 bas_gg[8])
{
    float16 du = 0e0f;
    
    for(int i=0; i<8; i++)
    {
        du.s048 += uu2[i].x*bas_gg[i];
        du.s159 += uu2[i].y*bas_gg[i];
        du.s26a += uu2[i].z*bas_gg[i];
        du.s37b += uu2[i].w*bas_gg[i];
    }
    return du;
}

//tensor basis gradient
float16 bas_tens(int dim, float3 g)
{
    float16 du = 0e0f;
    
    du.s048 = (dim==0)?g:0e0f;
    du.s159 = (dim==1)?g:0e0f;
    du.s26a = (dim==2)?g:0e0f;
    
    return du;
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
float3 mtx_mv(float16 A, float3 v)
{
    return v.x*A.s012 + v.y*A.s456 + v.z*A.s89a;
}

float16 mtx_mm(float16 A, float16 B)
{
    float16 C = 0e0f;
    
    C.s012 = mtx_mv(A,B.s012);
    C.s456 = mtx_mv(A,B.s456);
    C.s89a = mtx_mv(A,B.s89a);
    
    return C;
}


//outer prod
float16 mtx_uvT(float3 u, float3 v)
{
    return (float16){(float3)v.x*u,0e0f,(float3)v.y*u,0e0f,(float3)v.z*u,0e0f,0e0f,0e0f,0e0f,0e0f};
}

//transpose
float16 mtx_mT(float16 A)
{
    return A.s048c159d26ae37bf;
}

//matrix * diagonal
float16 mtx_md(float16 A, float3 D)
{
    return (float16){(float3)D.x*A.s012, 0e0f, (float3)D.y*A.s456, 0e0f, (float3)D.z*A.s89a, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
}


/*
 ===================================
 symmetric R^3x3
 ===================================
 */

//sym trace
float sym_tr(float8 A)
{
    return dot(A.s035,(float3){1e0f,1e0f,1e0f});
}


//sym determinant
float sym_det(float8 A)
{
    return dot((float3){A.s0,A.s1,A.s2}, cross((float3){A.s1, A.s3, A.s4}, (float3){A.s2, A.s4, A.s5}));
}


//outer product vv^T
float8 sym_vvT(float3 v)
{
    return (float8){v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z, 0e0f, 0e0f};
}

//sym mtx-vec
float3 sym_mv(float8 A, float3 v)
{
    return v.x*A.s012 + v.y*A.s134 + v.z*A.s245;
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


//mult V, diag D
float8  sym_mdmT(float16 V, float3 D)
{
    return (float8){D.x*V.s0*V.s0 + D.y*V.s4*V.s4 + D.z*V.s8*V.s8,
                    D.x*V.s0*V.s1 + D.y*V.s4*V.s5 + D.z*V.s8*V.s9,
                    D.x*V.s0*V.s2 + D.y*V.s4*V.s6 + D.z*V.s8*V.sa,
                    D.x*V.s1*V.s1 + D.y*V.s5*V.s5 + D.z*V.s9*V.s9,
                    D.x*V.s1*V.s2 + D.y*V.s5*V.s6 + D.z*V.s9*V.sa,
                    D.x*V.s2*V.s2 + D.y*V.s6*V.s6 + D.z*V.sa*V.sa, 0e0f, 0e0f};
}


//sum S = A+A^T
float8 sym_sumT(float16 A)
{
    float8 S = 0e0f;
    
    S.s012345 = A.s01256a + A.s04859a;

    return S;
}


//sym tensor inner prod
float sym_tip(float8 A, float8 B)
{
//    return A.s0*B.s0 + 2e0f*A.s1*B.s1 + 2e0f*A.s2*B.s2 + A.s3*B.s3 + 2e0f*A.s4*B.s4 + A.s5*B.s5;
    
    return dot(A.s0123,B.s0123) + dot(A.s45,B.s45);
}


/*
 ===================================
 mechanics
 ===================================
 */

//strain (du + du^T)/2
float8 mec_E(float16 du)
{
    return 5e-1f*sym_sumT(du);
}


//stress pk2 = lam*tr(e)*I + 2*mu*e
float8 mec_S(float8 E, float8 mat_prm)
{
    float8 S = 2e0f*mat_prm.s1*E;
    S.s035 += mat_prm.s0*sym_tr(E);
    
    return S;
}


//energy = 0.5*lam*(tr(E))^2 + mu*tr(E^2)
float mec_p(float8 E, float8 mat_prm)
{
    return 5e-1f*mat_prm.s0*pown(sym_tr(E),2) + mat_prm.s1*sym_tr(sym_mm(E,E));
}


//energy pos (miehe2010)
float mec_p1(float3 D, float8 mat_prm)
{
    //trace
    float trD = dot(D,(float3){1e0f,1e0f,1e0f});
    
    //split
    float3 D1;
    float3 D2;
    mec_D1D2(D, &D1, &D2);
    
    return 5e-1f*mat_prm.s0*trD*trD*(trD>0e0f) + mat_prm.s1*dot(D1,D1);
}


//split eigenvalues miehe2010
void mec_D1D2(float3 D, float3 *D1, float3 *D2)
{
    *D1 = 5e-1f*(fabs(D) + D);
    *D2 = 5e-1f*(fabs(D) - D);
    
    return;
}


//split stress (miehe2010)
void mec_S1S2(float3 D, float16 V, float8 mat_prm, float8 *S1, float8 *S2)
{
    float trE = dot(D,(float3){1e0f,1e0f,1e0f});
    
    //ramp
    trE = 5e-1f*(fabs(trE)+trE);
    
    //vals
    float3 D1 = 0e0f;
    float3 D2 = 0e0f;
    mec_D1D2(D, &D1, &D2);
    
    //vecs
    float8 vv1 = sym_vvT(V.s012);
    float8 vv2 = sym_vvT(V.s456);
    float8 vv3 = sym_vvT(V.s89a);
    
    //pos
    *S1 += (mat_prm.s0*trE + 2e0f*mat_prm.s1*D1.x)*vv1;
    *S1 += (mat_prm.s0*trE + 2e0f*mat_prm.s1*D1.y)*vv2;
    *S1 += (mat_prm.s0*trE + 2e0f*mat_prm.s1*D1.z)*vv3;
    
    //neg
    *S2 += (mat_prm.s0*trE + 2e0f*mat_prm.s1*D2.x)*vv1;
    *S2 += (mat_prm.s0*trE + 2e0f*mat_prm.s1*D2.y)*vv2;
    *S2 += (mat_prm.s0*trE + 2e0f*mat_prm.s1*D2.z)*vv3;
    
    return;
}


//derivative of A in direction of dA where A = VDV^T, Jodlbauer2020
void mec_dS1dS2(float8 dA, float3 D, float16 V, float8 *dS1, float8 *dS2)
{
    //vec derivs
    float16 dV;
    dV.s012 = -sym_mv(sym_mm(eig_mpinv(D.x, D, V), dA), V.s012);
    dV.s456 = -sym_mv(sym_mm(eig_mpinv(D.y, D, V), dA), V.s456);
    dV.s89a = -sym_mv(sym_mm(eig_mpinv(D.z, D, V), dA), V.s89a);

    //val derivs
    float3  dD;
    dD.x = dot(V.s012, sym_mv(dA, V.s012));
    dD.y = dot(V.s456, sym_mv(dA, V.s456));
    dD.z = dot(V.s89a, sym_mv(dA, V.s89a));
    
    //split derivs
    float3 dD1;
    float3 dD2;
    mec_D1D2(dD, &dD1, &dD2);
    
    //split vals
    float3 D1;
    float3 D2;
    mec_D1D2(D, &D1, &D2);
    
    //derivs
    *dS1 = sym_mdmT(V, dD1) + sym_sumT(mtx_mm(mtx_md(dV,D1), mtx_mT(V) ));
    *dS2 = sym_mdmT(V, dD2) + sym_sumT(mtx_mm(mtx_md(dV,D2), mtx_mT(V) ));

    return;
}


//derivative of energy wrt disp  d(energy+)/du
float mec_dp1(float3 D, float16 V, float16 dU, float8 mat_prm)
{
    //derivatives of principal strains wrt perturbation
    float3 dD = {dot(V.s012, mtx_mv(dU, V.s012)), dot(V.s456, mtx_mv(dU, V.s456)), dot(V.s89a, mtx_mv(dU, V.s89a))};

    //sum
    float sumdD = dot(dD,(float3){1e0f,1e0f,1e0f});
    
    //split vals
    float3 D1;
    float3 D2;
    mec_D1D2(D, &D1, &D2);
    
    //trace
    float trE = dot(D,(float3){1e0f,1e0f,1e0f});
    
    //test pos trace for first part, inidividual eigs (primary strains) for second part
    return mat_prm.s0*(trE>0e0f)*sumdD + 2e0f*mat_prm.s1*dot(D1,D1);
}



/*
 ===================================
 eigs (sym 3x3)
 ===================================
 */

//eigenvalues (A real symm) - Deledalle2017
float3 eig_val(float8 A)
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
    float3 D = 0e0f;
    D.x = (a + b + c - 2e0f*sqrt(x1)*cos((phi         )/3e0f))/3e0f;
    D.y = (a + b + c + 2e0f*sqrt(x1)*cos((phi - M_PI_F)/3e0f))/3e0f;
    D.z = (a + b + c + 2e0f*sqrt(x1)*cos((phi + M_PI_F)/3e0f))/3e0f;
    
    return D;
}


//eigenvectors (A real symm) - Kopp2008
float16 eig_vec(float8 A, float3 D)
{
    float16 V = 0e0f;
    
    //cross, normalise, skip when lam=0
    V.s012 = normalize(cross((float3){A.s0-D.x, A.s1, A.s2},(float3){A.s1, A.s3-D.x, A.s4}))*(D.x!=0e0f);
    V.s456 = normalize(cross((float3){A.s0-D.y, A.s1, A.s2},(float3){A.s1, A.s3-D.y, A.s4}))*(D.y!=0e0f);
    V.s89a = normalize(cross((float3){A.s0-D.z, A.s1, A.s2},(float3){A.s1, A.s3-D.z, A.s4}))*(D.z!=0e0f);
    
    return V;
}



//moore-penrose inverse
float8 eig_mpinv(float lam, float3 D, float16 V)
{
    //inv diag
    float3 Di;
    Di.x = (D.x==lam)?0e0f:1e0f/(D.x-lam);
    Di.y = (D.y==lam)?0e0f:1e0f/(D.y-lam);
    Di.z = (D.z==lam)?0e0f:1e0f/(D.z-lam);
    
    //pseudoinverse
    return sym_mdmT(V, Di);
}





////split direct from basis gradient and dim
//void eig_E1E2(float3 g, int dim, float8 *E1, float8 *E2)
//{
//    float n = vec_norm(g);
//
//    float3 g1 = vec_smulf(vec_saddf(g, -n), 5e-1f);
//    float3 g2 = vec_smulf(vec_saddf(g, +n), 5e-1f);
//
//    //vals (d2 is always zero)
//    float d0[3] = {g1.x, g1.y, g1.z};
//    float d1[3] = {g2.x, g2.y, g2.z};
//
//    //vecs
//    float3 v0[3];
//    v0[0] = vec_unit((float3){g1.x, g.y, g.z});
//    v0[1] = vec_unit((float3){g.x, g1.y, g.z});
//    v0[2] = vec_unit((float3){-g.x*g2.z, -g.y*g2.z, g.x*g.x + g.y*g.y});
//
//    float3 v1[3];
//    v1[0] = vec_unit((float3){g2.x, g.y, g.z});
//    v1[1] = vec_unit((float3){g.x, g2.y, g.z});
//    v1[2] = vec_unit((float3){-g.x*g1.z, -g.y*g1.z, g.x*g.x + g.y*g.y});
//
//    //select
//    *E1 = sym_smul(vec_out(v0[dim]),(d0[dim]>0e0f)*d0[dim]) + sym_smul(vec_out(v1[dim]),(d1[dim]>0e0f)*d1[dim]);
//    *E2 = sym_smul(vec_out(v0[dim]),(d0[dim]<0e0f)*d0[dim]) + sym_smul(vec_out(v1[dim]),(d1[dim]<0e0f)*d1[dim]);
//
//    return;
//}


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
    
    //vec
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
    
//    printf("vtx1_pos1 %v3d\n", vtx1_pos1);
    
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
                float16 du = bas_itpg(uu21, bas_gg);
                
//                printf("%+v3f\n", du.s048);
//                printf("%+v3f\n", du.s159);
//                printf("%+v3f\n", du.s26a);
//                printf("\n");
                
                //itp crack
                float c0 = itp_u0.w;
                float c1 = itp_u1.w;
                float dc = c1 - c0;                                             //for heaviside
                float gg[3] = {pown(1e0f - c1, 2), 2e0f*(c1 - 1e0f), 2e0f};     //pre-calc  gg derivs
                
                //strain
                float8 E = mec_E(du);
                float trE = sym_tr(E);
                
                //printf("%v8e\n", E); //ok
                
                //decompose E
                float3  D = eig_val(E);
                float16 V = eig_vec(E, D);
                
                
                //printf("%e %e %e\n", D[0],D[1],D[2]);
                //printf("%v3e %v3e %v3e\n", V[0],V[1],V[2]);
                
                //energy (pos)
                float p1 = mec_p1(D, mat_prm);
                
                //split stress
                float8 S1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                float8 S2 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                mec_S1S2(D, V, mat_prm, &S1, &S2);
                
                //                printf("%v8f\n",S1);
                //                printf("%v8f\n",S2);
                
                //local ptr
                global float* ff = (global float*)&F1[vtx1_idx1];
                
                //rhs u
                for(int dim1=0; dim1<3; dim1++)
                {
                    //tensor basis
                    float16 du1 = bas_tens(dim1, bas_gg[vtx1_idx2]);
                    
//                    printf("%+v3f\n", du1.s048);
//                    printf("%+v3f\n", du1.s159);
//                    printf("%+v3f\n", du1.s26a);
//                    printf("\n");
                    
                    //strain
                    float8 E1 = mec_E(du1);
                    
                    //printf("%+v8f\n", E1); //ok
                    
                    //write
                    ff[dim1] += sym_tip(gg[0]*S1 + S2, E1)*qw;
                }
                
                //rhs c
                ff[3] += ((gg[1]*p1 + mat_prm.s4*c1 + mat_prm.s6*(dc)*(dc<0e0f))*bas_ee[vtx1_idx2] + mat_prm.s5*dot(du[3], bas_gg[vtx1_idx2]))*qw;
                
                //vtx2
                for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
                {
                    //idx
                    int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
                    int  vtx2_idx3 = fn_idx3(vtx2_pos3);
                    
                    //local ptr
                    int idx1 = 27*vtx1_idx1 + vtx2_idx3;
                    global float* vv = (global float*)&J_vv[idx1];
                    
                    //cc write
                    vv[15] += ((gg[2]*p1 + mat_prm.s4 + mat_prm.s6*(dc<0e0f))*bas_ee[vtx1_idx2]*bas_ee[vtx2_idx2])*qw;
                    vv[15] += (mat_prm.s5*dot(bas_gg[vtx1_idx2], bas_gg[vtx2_idx2]))*qw;
                    
                    
                    //dim1
                    for(int dim1=0; dim1<3; dim1++)
                    {
                        //tensor basis
                        float16 du1 = bas_tens(dim1, bas_gg[vtx1_idx2]);
                        
                        //strain
                        float8 E1 = mec_E(du1);
                        
                        //idx
                        int idx3 = 4*dim1 + 3;
                        int idx4 = 12 + dim1;
                        
                        //coupling uc/cu
                        float cpl = gg[1]*mec_dp1(D, V, du1, mat_prm)*bas_ee[vtx1_idx2]*qw; //not sure about this yet
                        
                        //uc write
                        vv[idx3] += cpl;
                        vv[idx4] += cpl;
                        
                        //dim2
                        for(int dim2=0; dim2<3; dim2++)
                        {
                            //tensor basis
                            float16 du2 = bas_tens(dim1, bas_gg[vtx2_idx2]);
                            
                            //strain
                            float8 E2 = mec_E(du2);
                            float trE2 = sym_tr(E2);
                            
                            //split strain (deriv) jodlbauer2020
                            float8 dS1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                            float8 dS2 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                            mec_dS1dS2(E2, D, V, &dS1, &dS2);
                            

                            //stress (lam*tr(E)I + 2*mu*E, pos/neg) jodlbauer2020
                            dS1 = 2e0f*mat_prm.s1*dS1;
                            dS1.s035 += mat_prm.s0*(trE>=0e0f)*(trE2);      //if(trace of solution)then(trace of basis)
                            
                            dS2 = 2e0f*mat_prm.s1*dS2;
                            dS2.s035 += mat_prm.s0*(trE<=0e0f)*(trE2);
                            
                            //write uu
                            int idx2 = 4*dim1 + dim2;
                            vv[idx2] += (gg[0]*sym_tip(dS1, E1) + sym_tip(dS2, E1))*qw;

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
    //    int b1 = (vtx1_pos1.z == 0);                    //base
    int b2 = (vtx1_pos1.z == (vtx_dim.z - 1));      //top
    
    
    
    //init U
    U1[vtx1_idx1] = (float4){(float3)b2*mat_prm.s7, 0e0f};
    
    return;
}


//dirichlet
kernel void vtx_bnd3(const  int3    vtx_dim,
                     global float4  *F1,
                     global float16 *J_vv)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //bools
    int b1 = (vtx1_pos1.z == 0);                    //base
    int b2 = (vtx1_pos1.z == (vtx_dim.z - 1));      //top
    
    //I,F=0
    if((b1+b2)>0) //or
    {
        //rhs to zero (no step)
        F1[vtx1_idx1].xyz = (float3){0e0f, 0e0f, 0e0f};
        
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


//crack I
kernel void vtx_bnd4(const  int3    vtx_dim,
                     global float4  *F1,
                     global float16 *J_vv)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    
    //vtx2
    for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
    {
        int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
        int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);
        int  vtx2_bnd1 = fn_bnd1(vtx2_pos1, vtx_dim);
        
        //local pointer
        int idx1 = 27*vtx1_idx1 + vtx2_idx3;
        global float* vv = (global float*)&J_vv[idx1];
        
        //cc->I
        vv[15] = vtx2_bnd1*(vtx1_idx1==vtx2_idx1);
        
    } //vtx2
    
    return;
}



//newton step
kernel void vtx_step(const  int3    vtx_dim,
                     global float4 *U0,
                     global float4 *U1)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //step
    U1[vtx1_idx1] = U0[vtx1_idx1] - 0.5f*U1[vtx1_idx1];
    
    return;
}
