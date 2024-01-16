//
//  slv.h
//  frac2
//
//  Created by Toby Simpson on 12.01.24.
//

#ifndef slv_h
#define slv_h



void dsp_vec(DenseVector_Float v)
{
    for(int i=0; i<v.count; i++)
    {
        printf("%+e ", v.data[i]);
    }
    printf("\n\n");
    
    return;
}


//solve
int slv_mtx(struct msh_obj *msh, struct ocl_obj *ocl)
{
    printf("slv_u\n");
    
    //init mtx
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
    atts.transpose  = false;

    //size of input array
    long blk_num = 27*16*msh->nv_tot;
    int num_rows = 4*msh->nv_tot;
    int num_cols = 4*msh->nv_tot;

    //create
    SparseMatrix_Float A = SparseConvertFromCoordinate(num_rows, num_cols, blk_num, 1, atts, ocl->hst.J.ii, ocl->hst.J.jj, ocl->hst.J.vv);  //duplicates sum
    
    //vecs
    DenseVector_Float u;
    DenseVector_Float f;
    
    u.count = 4*msh->nv_tot;
    f.count = 4*msh->nv_tot;
    
    u.data = (float*)ocl->hst.U1;
    f.data = (float*)ocl->hst.F1;

    /*
     ========================
     solve
     ========================
     */
    
    //iterate
//    SparseSolve(SparseConjugateGradient(), A, f, u);    // SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
    SparseSolve(SparseGMRES(), A, f, u);
//    SparseSolve(SparseLSMR(), A, f, u); //minres - symmetric
    
    //QR
//    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);       //no
//    SparseSolve(QR, f , u);
//    SparseCleanup(QR);
    
    //clean
    SparseCleanup(A);

    return 0;
}


#endif /* slv_h */
