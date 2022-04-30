//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_LC_CALIB_GAUSSIANFUNCTION_H
#define CHECKERBOARD_LC_CALIB_GAUSSIANFUNCTION_H


#include <iostream>
#include <random>
#include <chrono>

using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using namespace Eigen;

class GaussianFunction
{
public:
    //---Single Variate
    // Eval the gaussian function at each x_i (i=1 to n) for the given mu, sigma.
    static VectorXd eval( VectorXd x, double mu, double sigma  );
    static double eval( double x, double mu, double sigma );
    static VectorXd linspace( double start_t, double end_t, int n );

    //---Multivariate
    /// Multivariate gaussian function
    /// @param x: k vector
    /// @param mu: mean vector. must be a k vector
    /// @param sigma: covariance matrix, must be kxk
    static double eval( const VectorXd x, const VectorXd mu, const MatrixXd sigma );

    /// Vectorized multivariate gaussian
    /// @param x: 3xn. n data points of each 3 dimensions. 3 can be in general any positve n
    /// @param mu: 3 vector
    /// @param sigma: covariance matrix. must be postive definite 3x3
    static VectorXd eval( const MatrixXd& x, const VectorXd mu, const MatrixXd sigma );

    /// Test for validity of covariance matrix. A valid covariance matrix is a symmetric
    /// matrix and all positive singular values. This can be tested with determinary being positive.
    static bool isValidCovarianceMatrix( const MatrixXd& A );


    /// sample mean
    /// @param x will be 3xn, 3dimensional (say) and n datapoints.
    /// @param returns 3 vector
    static VectorXd sample_mean( const MatrixXd& x );
    static double sample_mean( const VectorXd& x );

    /// sample covariance matrix
    /// @param x will be 3xn, 3dimensional (say) and n datapoints.
    /// @param returns 3x3 matrix
    static MatrixXd sample_covariance_matrix( const MatrixXd& x );
    static double sample_variance( const VectorXd& x ); //< please note, this is sigma^2.
};

#endif //CHECKERBOARD_LC_CALIB_GAUSSIANFUNCTION_H
