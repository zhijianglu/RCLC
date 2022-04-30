//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_LC_CALIB_GMMFIT_H
#define CHECKERBOARD_LC_CALIB_GMMFIT_H

// This class contains methods for fitting GMMs on data.
#include <iostream>
#include <vector>
using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "GaussianFunction.h"
#include "TermColor.h"


class GMMFit
{
public:
    /// Fit a GMM Model with EM Algorithm
    /// Closely follows the process described here: https://www.youtube.com/playlist?list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt
    ///     @param in_vec: The input vector
    ///     @param K : The number of gaussians
    ///     @param mu[input,output] : Initial guess of mu
    ///     @param sigma[input,output] : Initial guess of sigma
    static bool fit_1d( const VectorXd& in_vec, const int K, vector<double>& mu, vector<double>& sigma,bool show_debug_info );




    /// Fit a GMM Model with EM Algorithm
    ///     @param in_vec : 3xN. N data points each of 3 (say) dimensions
    //      @param K : Number of gaussians to fit.
    //      @param mu[input/output] : Initial guess of mu of the gaussians. len(mu) == K. each mu must be of 3 (say) dimensional
    //      @param sigma[input/out] : initial guess for sigma. len(sigma) == K. each sigma need to be a square matrix of dxd (say 3x3)
    static bool fit_multivariate( const MatrixXd& in_vec, const int K,
                                  vector<VectorXd>& mu, vector<MatrixXd>& sigma, VectorXd& priors );
};


#endif //CHECKERBOARD_LC_CALIB_GMMFIT_H
