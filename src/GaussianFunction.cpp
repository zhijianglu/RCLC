//
// Created by lab on 2021/12/8.
//

#include "GaussianFunction.h"

VectorXd GaussianFunction::eval( VectorXd x, double mu, double sigma  )
{
    assert( x.rows() > 0 );
    assert( sigma > 0 );
    // cout << "  allocate " << x.rows() << endl;
    VectorXd out = VectorXd::Zero( x.rows() );
    double factor = 1.0 / ( sigma * sqrt( 2.0 * M_PI ) );
    double sigma_sqr = sigma * sigma;

    for( int i=0 ; i<x.rows() ; i++ )
    {
        out(i) = exp( -1.0 / (2*sigma_sqr) * (x(i) - mu) * (x(i) - mu) );
    }
    out = factor * out;

    return out;
}

double GaussianFunction::eval( double x, double mu, double sigma )
{
    assert( sigma > 0 );
    double factor = 1.0 / ( sigma * sqrt( 2.0 * M_PI ) );
    double result = factor * exp( -1.0 / (2*sigma*sigma) * (x - mu) * (x - mu) );
    return result;

}



VectorXd GaussianFunction::linspace( double start_t, double end_t, int n )
{
    assert( start_t < end_t  && n > 0 );
    // VectorXd t = ArrayXd::LinSpaced( start_t, end_t, n).matrix();
    VectorXd t = VectorXd::Zero( n );

    double v = start_t;
    double delta = (end_t - start_t) / double( n-1 );
    for( int i=0 ; i<n ; i++ )
    {
        // cout << "t(" << i << ") = " << v << endl;
        t(i) = v;
        v+= delta;
    }
    // cout << "t=\n" <<  t.transpose() << endl;
    return t;

}


//------------ Multivariate ------------------//
double GaussianFunction::eval( const VectorXd x, const VectorXd mu, const MatrixXd sigma )
{
    int k = x.rows();
    assert( x.rows() > 0 );
    assert( x.rows() == mu.rows()  );
    assert( x.cols() == 1 && mu.cols() == 1 );
    assert( sigma.rows() == x.rows() && sigma.rows() == sigma.cols() );

    // is sigma a valid covariance matrix ==> only positive definite matrix is a valid sigma.
    //  ie. all eigen values need to be positive
    assert( isValidCovarianceMatrix(sigma) );

    double denom = pow(2.0 * M_PI, k ) * sigma.determinant() ;
    // cout << "denom = " << denom << endl;
    assert( denom > 0 );

    double u =  (x - mu ).transpose() * sigma.inverse() * (x- mu);
    return ( 1.0 / sqrt(denom) ) * exp( -0.5 * u ) ;

}


VectorXd GaussianFunction::eval( const MatrixXd& x, const VectorXd mu, const MatrixXd sigma )
{
    int n = x.cols(); // number of datapoints
    int k = x.rows(); //number of dimensions

    assert( n > 0 && k > 0 );
    assert( mu.rows() == k );
    assert( sigma.rows() == sigma.cols() && k == sigma.rows() );


    // is sigma a valid covariance matrix ==> only positive definite matrix is a valid sigma.
    //  ie. all eigen values need to be positive
    assert( isValidCovarianceMatrix(sigma) );

    double denom = sqrt( pow(2.0 * M_PI, k ) * sigma.determinant() );
    assert( denom > 0 );

    MatrixXd r = (x).colwise() - mu;
    VectorXd out = VectorXd::Zero( n );
    for( int i=0 ; i<n ; i++ )
    {
        double u = r.col(i).transpose() * sigma.inverse() * r.col(i);
        out( i ) = exp( -0.5 * u );
    }
    out /= denom;
    return out;
}


// #define __GaussianFunction__isValidCovarianceMatrix( msg ) msg;
#define __GaussianFunction__isValidCovarianceMatrix( msg ) ;
bool GaussianFunction::isValidCovarianceMatrix( const MatrixXd& A )
{
    if( A.rows() == 0 || A.cols() == 0 ) {
        __GaussianFunction__isValidCovarianceMatrix(
            cout << "[isValidCovarianceMatrix] returned false because either of rows or cols count is zero\n";
        )
        return false;
    }

    if( A.rows() != A.cols() ) {
        __GaussianFunction__isValidCovarianceMatrix(
            cout << "[isValidCovarianceMatrix] returned false because input matrix is not a square matrix\n";
        )
        return false;
    }

    // check about symmetry
    double diff = (A - A.transpose()).norm() ;
    if( diff > 1e-6 ) {
        __GaussianFunction__isValidCovarianceMatrix(
            cout << "[isValidCovarianceMatrix] returned false because `A - A.transpose()` is non zero, ie. input matrix is not symmetric.\n";
        )
        return false;
    }

    if( A.determinant() > 0 ) {
        __GaussianFunction__isValidCovarianceMatrix(
            cout << "determinant is positive. WARN: this is not a complete test, need to look at eigen values and make sure they are all positives\n";
        )

        //  need more testing before declaring it as OK for covariance matrix. Eigen Value test
        EigenSolver<MatrixXd> es(A, true);
        VectorXcd eigs = es.eigenvalues();
        for( int i=0 ; i<eigs.rows() ; i++ ) {
            double __o =(eigs(i,0)).real() ;
            __GaussianFunction__isValidCovarianceMatrix(cout << i << "th eigen value = " << __o << endl;)
            if( __o <= 0 ) {
                __GaussianFunction__isValidCovarianceMatrix(
                    cout << "return false, found an eigen value which was negative. So this cannot be a covariance matrix (positive definite)\n";)
                return false;
            }
        }
        __GaussianFunction__isValidCovarianceMatrix(
            cout << "return true, All the eigen values are postive\n";)
        return true;
    }else {
        __GaussianFunction__isValidCovarianceMatrix(
            cout << "Return false because determinant is negative. This means some eigen values must be negative, hence this cannot be a covariance matrix\n";
        )
        return false;
    }



#if 0
    // Using SVD
    JacobiSVD<MatrixXd> svd( A, ComputeFullV | ComputeFullU );
    VectorXd singular_values = svd.singularValues();
    for( int i=0 ; i<singular_values.rows() ; i++ )
    {
        if( singular_values(i) <= 0 ) {
            __GaussianFunction__isValidCovarianceMatrix(
            cout << "[isValidCovarianceMatrix] returned false because i found a non positive singular value\n";
            )
            return false;
        }
    }

    __GaussianFunction__isValidCovarianceMatrix(
    cout << "[isValidCovarianceMatrix] return true, this matrix appear to be a covariance matrix\n";)
    return true;
#endif




}




VectorXd GaussianFunction::sample_mean( const MatrixXd& x )
{
    int d = x.rows();  //data dimensions
    int n = x.cols() ; //number of samples.
    assert( d>0 && n > 0 );
    return x.rowwise().mean();
}


double GaussianFunction::sample_mean( const VectorXd& x )
{
    assert( x.rows() > 0 );
    return x.mean();
}


MatrixXd GaussianFunction::sample_covariance_matrix( const MatrixXd& x )
{
    int d = x.rows();  //data dimensions
    int n = x.cols() ; //number of samples.
    VectorXd sample_mu = sample_mean( x );
    assert( sample_mu.rows() == d );
    return 1.0 / double(n-1) * (x.colwise() - sample_mu)  * (x.colwise() - sample_mu).transpose();
}

double GaussianFunction::sample_variance( const VectorXd& x )
{
    double s_mu = sample_mean( x );
    int n = x.rows();
    assert( n > 1 && "cannot comute variance with 1 sample");
    double at_a = x.transpose() * x;
    return 1.0 / double(n-1) * ( at_a - s_mu*s_mu);
}